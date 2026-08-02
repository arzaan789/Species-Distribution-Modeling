"""Result-blind pilot and restartable flexible-model sensitivity runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_sdm.backgrounds import make_backgrounds
from provenance_sdm.config import StudyConfig
from provenance_sdm.flexible_maxent import (
    FLEXIBLE_FEATURE_BASIS,
    fit_flexible_maxent,
)
from provenance_sdm.landscape import Landscape
from provenance_sdm.manifests import write_manifest
from provenance_sdm.mechanism_runner import (
    _atomic_keyed_parquet,
    _evaluation_hash,
    _key_set,
    _missing_unexpected,
)
from provenance_sdm.metrics import generate_unbiased_evaluation, truth_metrics
from provenance_sdm.observation import simulate_observations, simulate_programmes
from provenance_sdm.provenance import source_distribution_distance
from provenance_sdm.simulation_runner import (
    PRIMARY_METRICS,
    STABILITY_COLUMNS,
    _landscape_hash,
    seed_for,
)
from provenance_sdm.virtual_species import simulate_species_truth


FLEXIBLE_ARMS = ("conventional_tgb", "pm_tgb")
FLEXIBLE_KEY_COLUMNS = (
    "community_seed",
    "alignment",
    "bias_level",
    "species_id",
    "background_arm",
)
PILOT_KEY_COLUMNS = (*FLEXIBLE_KEY_COLUMNS, "model_regularization")


def _subset_species_ids(n_species: int) -> tuple[str, ...]:
    return tuple(f"sp_{index:03d}" for index in range(0, n_species, 4))


def expected_flexible_keys(
    config: StudyConfig,
    community_indices: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """Return exact keys for the deterministic every-fourth-species subset."""

    simulation = config.simulation
    rows = []
    for community_index in community_indices:
        if community_index < 0 or community_index >= simulation.n_communities:
            continue
        community_seed = seed_for(simulation.seed, "community", community_index)
        for alignment in simulation.alignments:
            for bias_level in simulation.bias_levels:
                for species_id in _subset_species_ids(simulation.n_species):
                    for arm in FLEXIBLE_ARMS:
                        rows.append(
                            (
                                community_seed,
                                alignment,
                                bias_level,
                                species_id,
                                arm,
                            )
                        )
    return pd.DataFrame(rows, columns=FLEXIBLE_KEY_COLUMNS)


def _stable_rows(rows: pd.DataFrame) -> bool:
    return bool(
        not rows.empty
        and all(column in rows for column in STABILITY_COLUMNS)
        and np.isfinite(rows.loc[:, STABILITY_COLUMNS].to_numpy(dtype=float)).all()
        and "fit_succeeded" in rows
        and rows.fit_succeeded.eq(True).all()
        and rows.max_cell_mass.le(0.10).all()
        and rows.effective_cell_count.ge(50.0).all()
        and rows.lower_clip_fraction.between(0.0, 1.0).all()
        and "solver_converged" in rows
        and rows.solver_converged.eq(True).all()
    )


def select_flexible_regularization(
    pilot_rows: pd.DataFrame,
    regularizations: Sequence[float],
) -> dict[str, object]:
    """Select the first candidate using stability diagnostics only."""

    candidates = [float(value) for value in regularizations]
    for candidate in candidates:
        rows = pilot_rows.query("model_regularization == @candidate")
        if _stable_rows(rows):
            return {
                "include": True,
                "regularization": candidate,
                "reason": "smallest fully stable candidate",
                "regularizations": candidates,
            }
    return {
        "include": False,
        "regularization": None,
        "reason": "no candidate passed every stability check",
        "regularizations": candidates,
    }


def _candidate_values(regularizations: Sequence[float]) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in regularizations)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or not np.isfinite(candidates).all()
        or any(value <= 0 for value in candidates)
        or tuple(sorted(candidates)) != candidates
    ):
        raise ValueError(
            "regularizations must be unique positive finite values in ascending order"
        )
    return candidates


def _scenario(
    config: StudyConfig,
    landscape: Landscape,
    community_index: int,
    alignment: str,
    bias_level: str,
):
    simulation = config.simulation
    community_seed = seed_for(simulation.seed, "community", community_index)
    truth = simulate_species_truth(
        landscape,
        simulation.n_species,
        seed_for(simulation.seed, "truth", community_seed),
    )
    programmes = simulate_programmes(
        landscape,
        simulation.n_programmes,
        bias_level,
        seed_for(simulation.seed, "programmes", community_seed, bias_level),
    )
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
    return community_seed, truth, observed


def _backgrounds_and_presence(
    config: StudyConfig,
    landscape: Landscape,
    observed,
    species,
    community_seed: int,
    alignment: str,
    bias_level: str,
):
    simulation = config.simulation
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
    return backgrounds, presence


def run_flexible_pilot(
    config: StudyConfig,
    landscape: Landscape,
    output_dir: Path,
    regularizations: Sequence[float] = (2.0, 5.0, 10.0),
) -> tuple[Path, Path]:
    """Run stability-only community-zero fits and write a deterministic gate."""

    candidates = _candidate_values(regularizations)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pilot_path = destination / "flexible_pilot.parquet"
    gate_path = destination / "flexible_gate.json"
    pilot = pd.read_parquet(pilot_path) if pilot_path.exists() else pd.DataFrame()
    landscape_hash = _landscape_hash(landscape)
    if not pilot.empty:
        if pilot.duplicated(list(PILOT_KEY_COLUMNS)).any():
            raise ValueError("existing flexible pilot contains duplicate keys")
        if not pilot.landscape_hash.eq(landscape_hash).all():
            raise ValueError("existing flexible pilot uses a different landscape")
        if not pilot.feature_basis.eq(FLEXIBLE_FEATURE_BASIS).all():
            raise ValueError("existing flexible pilot uses a different feature basis")

    expected = expected_flexible_keys(config, community_indices=(0,))
    expected_keys = _key_set(expected, FLEXIBLE_KEY_COLUMNS)
    selected_ids = set(_subset_species_ids(config.simulation.n_species))
    tested: list[float] = []
    for candidate in candidates:
        candidate_rows = (
            pilot.loc[pilot.model_regularization.eq(candidate)]
            if "model_regularization" in pilot
            else pd.DataFrame()
        )
        existing_keys = _key_set(candidate_rows, FLEXIBLE_KEY_COLUMNS)
        unexpected = existing_keys.difference(expected_keys)
        if unexpected:
            raise ValueError("existing flexible pilot contains unexpected keys")
        missing = expected_keys.difference(existing_keys)
        if missing:
            for alignment in config.simulation.alignments:
                for bias_level in config.simulation.bias_levels:
                    community_seed, truth, observed = _scenario(
                        config,
                        landscape,
                        0,
                        alignment,
                        bias_level,
                    )
                    for species in truth:
                        if species.species_id not in selected_ids:
                            continue
                        backgrounds, presence = _backgrounds_and_presence(
                            config,
                            landscape,
                            observed,
                            species,
                            community_seed,
                            alignment,
                            bias_level,
                        )
                        for arm in FLEXIBLE_ARMS:
                            key = (
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                                arm,
                            )
                            if key not in missing:
                                continue
                            row: dict[str, object] = {
                                "community_seed": community_seed,
                                "alignment": alignment,
                                "bias_level": bias_level,
                                "species_id": species.species_id,
                                "background_arm": arm,
                                "model_regularization": candidate,
                                "background_cells": len(backgrounds[arm]),
                                "landscape_hash": landscape_hash,
                                "feature_basis": FLEXIBLE_FEATURE_BASIS,
                            }
                            try:
                                model = fit_flexible_maxent(
                                    presence,
                                    backgrounds[arm],
                                    landscape.feature_names,
                                    candidate,
                                    seed_for(
                                        config.simulation.seed,
                                        "flexible-model",
                                        community_seed,
                                        alignment,
                                        bias_level,
                                        species.species_id,
                                        arm,
                                    ),
                                )
                                result = model.predict_with_diagnostics(landscape)
                                row.update(
                                    {
                                        "fit_succeeded": True,
                                        "max_cell_mass": result.max_cell_mass,
                                        "effective_cell_count": result.effective_cell_count,
                                        "log_intensity_range": result.log_intensity_range,
                                        "lower_clip_cells": result.lower_clip_cells,
                                        "lower_clip_fraction": result.lower_clip_fraction,
                                        "solver_converged": result.solver_converged,
                                        "failure_type": None,
                                        "failure_message": None,
                                    }
                                )
                            except Exception as exc:
                                row.update(
                                    {
                                        "fit_succeeded": False,
                                        "max_cell_mass": np.nan,
                                        "effective_cell_count": np.nan,
                                        "log_intensity_range": np.nan,
                                        "lower_clip_cells": np.nan,
                                        "lower_clip_fraction": np.nan,
                                        "solver_converged": False,
                                        "failure_type": type(exc).__name__,
                                        "failure_message": str(exc),
                                    }
                                )
                            pilot = pd.concat(
                                [pilot, pd.DataFrame([row])],
                                ignore_index=True,
                            )
                            _atomic_keyed_parquet(
                                pilot,
                                pilot_path,
                                PILOT_KEY_COLUMNS,
                            )
                            missing.remove(key)
        tested.append(candidate)
        candidate_rows = pilot.query("model_regularization == @candidate")
        if len(candidate_rows) == len(expected) and _stable_rows(candidate_rows):
            break

    gate = select_flexible_regularization(pilot, candidates)
    gate["tested_regularizations"] = tested
    gate["expected_rows_per_candidate"] = int(len(expected))
    gate["landscape_hash"] = landscape_hash
    gate["feature_basis"] = FLEXIBLE_FEATURE_BASIS
    write_manifest(gate, gate_path, allow_replace=True)
    return pilot_path, gate_path


def run_flexible_sensitivity(
    config: StudyConfig,
    landscape: Landscape,
    gate_path: Path,
    output_dir: Path,
) -> Path:
    """Run all exact flexible sensitivity keys after the stability gate."""

    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if gate.get("include") is not True:
        raise ValueError("flexible sensitivity was excluded by its stability gate")
    regularization = float(gate["regularization"])
    if regularization not in [float(value) for value in gate["regularizations"]]:
        raise ValueError("flexible gate selected an undeclared regularization")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "flexible_sensitivity_metrics.parquet"
    failure_path = destination / "flexible_failures.csv"
    completed = pd.read_parquet(result_path) if result_path.exists() else pd.DataFrame()
    expected = expected_flexible_keys(
        config,
        community_indices=tuple(range(config.simulation.n_communities)),
    )
    expected_keys = _key_set(expected, FLEXIBLE_KEY_COLUMNS)
    landscape_hash = _landscape_hash(landscape)
    if not completed.empty:
        if completed.duplicated(list(FLEXIBLE_KEY_COLUMNS)).any():
            raise ValueError("existing flexible results contain duplicate keys")
        if not _key_set(completed, FLEXIBLE_KEY_COLUMNS) <= expected_keys:
            raise ValueError("existing flexible results contain unexpected keys")
        if not completed.landscape_hash.eq(landscape_hash).all():
            raise ValueError("existing flexible results use a different landscape")
        if not completed.feature_basis.eq(FLEXIBLE_FEATURE_BASIS).all():
            raise ValueError("existing flexible results use a different feature basis")
        if not completed.model_regularization.eq(regularization).all():
            raise ValueError("existing flexible results use a different regularization")
    missing = expected_keys.difference(_key_set(completed, FLEXIBLE_KEY_COLUMNS))
    if not missing:
        failure_path.unlink(missing_ok=True)
        return result_path

    selected_ids = set(_subset_species_ids(config.simulation.n_species))
    failures: list[dict[str, object]] = []
    for community_index in range(config.simulation.n_communities):
        for alignment in config.simulation.alignments:
            for bias_level in config.simulation.bias_levels:
                community_seed, truth, observed = _scenario(
                    config,
                    landscape,
                    community_index,
                    alignment,
                    bias_level,
                )
                for species in truth:
                    if species.species_id not in selected_ids:
                        continue
                    requested = [
                        arm
                        for arm in FLEXIBLE_ARMS
                        if (
                            community_seed,
                            alignment,
                            bias_level,
                            species.species_id,
                            arm,
                        ) in missing
                    ]
                    if not requested:
                        continue
                    try:
                        backgrounds, presence = _backgrounds_and_presence(
                            config,
                            landscape,
                            observed,
                            species,
                            community_seed,
                            alignment,
                            bias_level,
                        )
                        focal_sources = observed.records.query(
                            "species_id == @species.species_id"
                        ).programme_id
                        candidate_sources = observed.records.query(
                            "taxonomic_group == @species.taxonomic_group "
                            "and species_id != @species.species_id"
                        ).programme_id
                        source_distance = source_distribution_distance(
                            focal_sources,
                            candidate_sources,
                        )
                        evaluation = generate_unbiased_evaluation(
                            species,
                            500,
                            500,
                            seed_for(
                                config.simulation.seed,
                                "evaluation",
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                            ),
                        )
                        evaluation_hash = _evaluation_hash(evaluation)
                        for arm in requested:
                            model = fit_flexible_maxent(
                                presence,
                                backgrounds[arm],
                                landscape.feature_names,
                                regularization,
                                seed_for(
                                    config.simulation.seed,
                                    "flexible-model",
                                    community_seed,
                                    alignment,
                                    bias_level,
                                    species.species_id,
                                    arm,
                                ),
                            )
                            result = model.predict_with_diagnostics(landscape)
                            prediction = result.suitability
                            metrics = truth_metrics(
                                prediction,
                                species,
                                evaluation.label.to_numpy(),
                                prediction[evaluation.cell_id.to_numpy(dtype=int)],
                            )
                            row = {
                                "community_seed": community_seed,
                                "alignment": alignment,
                                "bias_level": bias_level,
                                "species_id": species.species_id,
                                "background_arm": arm,
                                "record_count": len(presence),
                                "niche_breadth": float(species.niche_breadth),
                                "background_cells": len(backgrounds[arm]),
                                "source_distribution_distance": source_distance,
                                "unsupported_mass": (
                                    float(backgrounds[arm].unsupported_mass.iloc[0])
                                    if arm == "pm_tgb"
                                    else 0.0
                                ),
                                "landscape_hash": landscape_hash,
                                "evaluation_hash": evaluation_hash,
                                "feature_basis": result.feature_basis,
                                "model_regularization": regularization,
                                "max_cell_mass": result.max_cell_mass,
                                "effective_cell_count": result.effective_cell_count,
                                "log_intensity_range": result.log_intensity_range,
                                "lower_clip_cells": result.lower_clip_cells,
                                "lower_clip_fraction": result.lower_clip_fraction,
                                "solver_converged": result.solver_converged,
                                **metrics,
                            }
                            completed = pd.concat(
                                [completed, pd.DataFrame([row])],
                                ignore_index=True,
                            )
                            _atomic_keyed_parquet(
                                completed,
                                result_path,
                                FLEXIBLE_KEY_COLUMNS,
                            )
                            missing.remove(
                                (
                                    community_seed,
                                    alignment,
                                    bias_level,
                                    species.species_id,
                                    arm,
                                )
                            )
                    except Exception as exc:
                        failures.extend(
                            {
                                "community_seed": community_seed,
                                "alignment": alignment,
                                "bias_level": bias_level,
                                "species_id": species.species_id,
                                "background_arm": arm,
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                            }
                            for arm in requested
                        )
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    else:
        failure_path.unlink(missing_ok=True)
    return result_path


def audit_flexible_sensitivity(
    output_dir: Path,
    config: StudyConfig,
) -> dict[str, object]:
    """Audit the result-blind gate and any included full sensitivity run."""

    destination = Path(output_dir)
    gate = json.loads(
        (destination / "flexible_gate.json").read_text(encoding="utf-8")
    )
    pilot = pd.read_parquet(destination / "flexible_pilot.parquet")
    expected_pilot = expected_flexible_keys(config, community_indices=(0,))
    tested = [float(value) for value in gate.get("tested_regularizations", [])]
    pilot_complete = True
    pilot_stability_matches = True
    for candidate in tested:
        rows = pilot.query("model_regularization == @candidate")
        missing, unexpected = _missing_unexpected(
            rows,
            expected_pilot,
            FLEXIBLE_KEY_COLUMNS,
        )
        pilot_complete &= missing.empty and unexpected.empty and len(rows) == len(expected_pilot)
        if gate.get("include") is True and candidate == float(gate["regularization"]):
            pilot_stability_matches &= _stable_rows(rows)
    pilot_duplicates = int(pilot.duplicated(list(PILOT_KEY_COLUMNS)).sum())
    pilot_labels = bool(
        not pilot.empty
        and pilot.feature_basis.eq(FLEXIBLE_FEATURE_BASIS).all()
        and pilot.model_regularization.isin(tested).all()
        and not set(PRIMARY_METRICS).intersection(pilot.columns)
    )
    included = gate.get("include") is True
    full_complete = not included
    full_stable = not included
    full_finite = not included
    full_labels = not included
    completed_full = 0
    missing_full = 0
    unexpected_full = 0
    duplicate_full = 0
    if included:
        result_path = destination / "flexible_sensitivity_metrics.parquet"
        if result_path.is_file():
            full = pd.read_parquet(result_path)
            expected_full = expected_flexible_keys(
                config,
                community_indices=tuple(range(config.simulation.n_communities)),
            )
            missing, unexpected = _missing_unexpected(
                full,
                expected_full,
                FLEXIBLE_KEY_COLUMNS,
            )
            completed_full = int(len(full))
            missing_full = int(len(missing))
            unexpected_full = int(len(unexpected))
            duplicate_full = int(full.duplicated(list(FLEXIBLE_KEY_COLUMNS)).sum())
            full_complete = bool(
                missing.empty
                and unexpected.empty
                and duplicate_full == 0
                and len(full) == len(expected_full)
            )
            full_stable = _stable_rows(full.assign(fit_succeeded=True))
            full_finite = bool(
                all(column in full for column in PRIMARY_METRICS)
                and np.isfinite(full.loc[:, PRIMARY_METRICS].to_numpy(dtype=float)).all()
            )
            full_labels = bool(
                full.feature_basis.eq(FLEXIBLE_FEATURE_BASIS).all()
                and full.model_regularization.eq(float(gate["regularization"])).all()
                and full.evaluation_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
            )
        else:
            full_complete = full_stable = full_finite = full_labels = False
    else:
        declared = [float(value) for value in gate.get("regularizations", [])]
        pilot_complete &= tested == declared
        pilot_stability_matches &= all(
            not _stable_rows(pilot.query("model_regularization == @candidate"))
            for candidate in tested
        )
    failure_path = destination / "flexible_failures.csv"
    failures = len(pd.read_csv(failure_path)) if failure_path.exists() else 0
    passed = bool(
        tested
        and pilot_complete
        and pilot_stability_matches
        and pilot_duplicates == 0
        and pilot_labels
        and full_complete
        and full_stable
        and full_finite
        and full_labels
        and failures == 0
    )
    return {
        "status": "passed" if passed else "failed",
        "full_run_included": bool(included),
        "selected_regularization": (
            float(gate["regularization"]) if included else None
        ),
        "tested_regularizations": tested,
        "pilot_complete": bool(pilot_complete),
        "pilot_stability_matches_gate": bool(pilot_stability_matches),
        "pilot_duplicates": pilot_duplicates,
        "pilot_labels_valid": pilot_labels,
        "completed_full": completed_full,
        "missing_full": missing_full,
        "unexpected_full": unexpected_full,
        "duplicate_full": duplicate_full,
        "full_complete": bool(full_complete),
        "full_stable": bool(full_stable),
        "full_finite": bool(full_finite),
        "full_labels_valid": bool(full_labels),
        "failures": int(failures),
        "gate_reason": str(gate.get("reason", "")),
        "configuration_hash": hashlib.sha256(
            repr(config).encode("utf-8")
        ).hexdigest(),
    }
