"""Restartable source-composition diagnostics and latent-mixture fits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from provenance_sdm.backgrounds import (
    make_backgrounds,
    make_latent_mixture_background,
)
from provenance_sdm.config import StudyConfig
from provenance_sdm.landscape import Landscape
from provenance_sdm.maxent import PRIMARY_REGULARIZATION, fit_maxent
from provenance_sdm.mechanism import mechanism_row
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


MECHANISM_KEY_COLUMNS = (
    "community_seed",
    "alignment",
    "bias_level",
    "species_id",
)
LATENT_KEY_COLUMNS = (*MECHANISM_KEY_COLUMNS, "background_arm")
LATENT_ARM = "latent_mixture_tgb"


def expected_mechanism_keys(config: StudyConfig) -> pd.DataFrame:
    """Return the frozen key matrix for all species-scenario diagnostics."""

    rows = []
    simulation = config.simulation
    for community_index in range(simulation.n_communities):
        community_seed = seed_for(simulation.seed, "community", community_index)
        for alignment in simulation.alignments:
            for bias_level in simulation.bias_levels:
                for species_index in range(simulation.n_species):
                    rows.append(
                        (
                            community_seed,
                            alignment,
                            bias_level,
                            f"sp_{species_index:03d}",
                        )
                    )
    return pd.DataFrame(rows, columns=MECHANISM_KEY_COLUMNS)


def _atomic_keyed_parquet(
    rows: pd.DataFrame,
    path: Path,
    key_columns: tuple[str, ...],
) -> None:
    if rows.duplicated(list(key_columns)).any():
        raise ValueError(f"{path.name} contains duplicate keys")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    rows.to_parquet(temporary, index=False)
    check = pd.read_parquet(temporary)
    if len(check) != len(rows) or check.duplicated(list(key_columns)).any():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{path.name} failed atomic checkpoint validation")
    os.replace(temporary, path)


def _key_set(rows: pd.DataFrame, columns: tuple[str, ...]) -> set[tuple[object, ...]]:
    if rows.empty:
        return set()
    return set(
        rows.loc[:, columns].itertuples(index=False, name=None)
    )


def _evaluation_hash(evaluation: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        evaluation.loc[:, ["cell_id", "label"]],
        index=False,
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_existing(
    diagnostics: pd.DataFrame,
    latent: pd.DataFrame,
    expected: pd.DataFrame,
    landscape_hash: str,
) -> None:
    expected_keys = _key_set(expected, MECHANISM_KEY_COLUMNS)
    for rows, columns, diagnostic in (
        (diagnostics, MECHANISM_KEY_COLUMNS, "mechanism diagnostics"),
        (latent, LATENT_KEY_COLUMNS, "latent-mixture metrics"),
    ):
        if rows.empty:
            continue
        if rows.duplicated(list(columns)).any():
            raise ValueError(f"existing {diagnostic} contain duplicate keys")
        keys = _key_set(rows, columns)
        reduced = {key[:4] for key in keys}
        if not reduced <= expected_keys:
            raise ValueError(f"existing {diagnostic} contain unexpected keys")
        if "landscape_hash" not in rows or not rows.landscape_hash.eq(landscape_hash).all():
            raise ValueError(f"existing {diagnostic} use a different landscape")
    if not latent.empty:
        if not latent.background_arm.eq(LATENT_ARM).all():
            raise ValueError("existing latent checkpoint uses a different arm")
        if "feature_basis" not in latent or not latent.feature_basis.eq("linear").all():
            raise ValueError("existing latent checkpoint uses a different feature basis")
        if (
            "model_regularization" not in latent
            or not latent.model_regularization.eq(PRIMARY_REGULARIZATION).all()
        ):
            raise ValueError(
                "existing latent checkpoint uses a different model regularization"
            )


def run_mechanism(
    config: StudyConfig,
    landscape: Landscape,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write missing mechanism rows and latent-mixture model results."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics_path = destination / "mechanism_diagnostics.parquet"
    latent_path = destination / "latent_mixture_metrics.parquet"
    failure_path = destination / "mechanism_failures.csv"
    diagnostics = (
        pd.read_parquet(diagnostics_path)
        if diagnostics_path.exists()
        else pd.DataFrame()
    )
    latent = pd.read_parquet(latent_path) if latent_path.exists() else pd.DataFrame()
    expected = expected_mechanism_keys(config)
    landscape_hash = _landscape_hash(landscape)
    _validate_existing(diagnostics, latent, expected, landscape_hash)
    expected_keys = _key_set(expected, MECHANISM_KEY_COLUMNS)
    missing_diagnostics = expected_keys.difference(
        _key_set(diagnostics, MECHANISM_KEY_COLUMNS)
    )
    missing_latent = expected_keys.difference(
        {key[:4] for key in _key_set(latent, LATENT_KEY_COLUMNS)}
    )
    if not missing_diagnostics and not missing_latent:
        failure_path.unlink(missing_ok=True)
        return diagnostics_path, latent_path

    failures: list[dict[str, object]] = []
    simulation = config.simulation
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
                scenario_prefix = (community_seed, alignment, bias_level)
                if not any(
                    key[:3] == scenario_prefix
                    for key in missing_diagnostics | missing_latent
                ):
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
                    key = (*scenario_prefix, species.species_id)
                    if key in missing_diagnostics:
                        candidate_sources = observed.records.query(
                            "taxonomic_group == @species.taxonomic_group "
                            "and species_id != @species.species_id"
                        ).programme_id
                        focal_sources = observed.records.query(
                            "species_id == @species.species_id"
                        ).programme_id
                        row = {
                            "community_seed": community_seed,
                            "alignment": alignment,
                            "bias_level": bias_level,
                            "species_id": species.species_id,
                            "source_distribution_distance": (
                                source_distribution_distance(
                                    focal_sources,
                                    candidate_sources,
                                )
                            ),
                            "landscape_hash": landscape_hash,
                            **mechanism_row(observed, species.species_id),
                        }
                        diagnostics = pd.concat(
                            [diagnostics, pd.DataFrame([row])],
                            ignore_index=True,
                        )
                        _atomic_keyed_parquet(
                            diagnostics,
                            diagnostics_path,
                            MECHANISM_KEY_COLUMNS,
                        )
                        missing_diagnostics.remove(key)
                    if key not in missing_latent:
                        continue
                    try:
                        primary_backgrounds = make_backgrounds(
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
                        latent_background = make_latent_mixture_background(
                            observed,
                            species.species_id,
                            len(primary_backgrounds["pm_tgb"]),
                            seed_for(
                                simulation.seed,
                                "latent-background",
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                            ),
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
                        model = fit_maxent(
                            presence,
                            latent_background,
                            landscape.feature_names,
                            regularization=PRIMARY_REGULARIZATION,
                            seed=seed_for(
                                simulation.seed,
                                "model",
                                community_seed,
                                alignment,
                                bias_level,
                                species.species_id,
                                LATENT_ARM,
                            ),
                        )
                        prediction_result = model.predict_with_diagnostics(landscape)
                        prediction = prediction_result.suitability
                        metrics = truth_metrics(
                            prediction,
                            species,
                            evaluation.label.to_numpy(),
                            prediction[evaluation.cell_id.to_numpy(dtype=int)],
                        )
                        latent_row = {
                            "community_seed": community_seed,
                            "alignment": alignment,
                            "bias_level": bias_level,
                            "species_id": species.species_id,
                            "background_arm": LATENT_ARM,
                            "record_count": len(presence),
                            "niche_breadth": float(species.niche_breadth),
                            "background_cells": len(latent_background),
                            "unsupported_mass": float(
                                latent_background.unsupported_mass.iloc[0]
                            ),
                            "landscape_hash": landscape_hash,
                            "evaluation_hash": _evaluation_hash(evaluation),
                            "feature_basis": prediction_result.feature_basis,
                            "model_regularization": PRIMARY_REGULARIZATION,
                            "max_cell_mass": prediction_result.max_cell_mass,
                            "effective_cell_count": (
                                prediction_result.effective_cell_count
                            ),
                            "log_intensity_range": (
                                prediction_result.log_intensity_range
                            ),
                            "lower_clip_cells": prediction_result.lower_clip_cells,
                            "lower_clip_fraction": (
                                prediction_result.lower_clip_fraction
                            ),
                            "solver_converged": prediction_result.solver_converged,
                            **metrics,
                        }
                        latent = pd.concat(
                            [latent, pd.DataFrame([latent_row])],
                            ignore_index=True,
                        )
                        _atomic_keyed_parquet(
                            latent,
                            latent_path,
                            LATENT_KEY_COLUMNS,
                        )
                        missing_latent.remove(key)
                    except Exception as exc:
                        failures.append(
                            {
                                "community_seed": community_seed,
                                "alignment": alignment,
                                "bias_level": bias_level,
                                "species_id": species.species_id,
                                "background_arm": LATENT_ARM,
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    else:
        failure_path.unlink(missing_ok=True)
    return diagnostics_path, latent_path


def _missing_unexpected(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_keys = actual.loc[:, columns].drop_duplicates()
    missing = expected.merge(
        actual_keys,
        on=list(columns),
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'").drop(columns="_merge")
    unexpected = actual_keys.merge(
        expected,
        on=list(columns),
        how="left",
        indicator=True,
    ).query("_merge == 'left_only'").drop(columns="_merge")
    return missing, unexpected


def audit_mechanism(
    output_dir: Path,
    config: StudyConfig,
) -> dict[str, object]:
    """Validate exact diagnostics and latent-mixture result artifacts."""

    destination = Path(output_dir)
    diagnostics = pd.read_parquet(destination / "mechanism_diagnostics.parquet")
    latent = pd.read_parquet(destination / "latent_mixture_metrics.parquet")
    expected_diagnostics = expected_mechanism_keys(config)
    expected_latent = expected_diagnostics.copy()
    expected_latent["background_arm"] = LATENT_ARM
    missing_diagnostics, unexpected_diagnostics = _missing_unexpected(
        diagnostics,
        expected_diagnostics,
        MECHANISM_KEY_COLUMNS,
    )
    missing_latent, unexpected_latent = _missing_unexpected(
        latent,
        expected_latent,
        LATENT_KEY_COLUMNS,
    )
    diagnostics_duplicates = int(
        diagnostics.duplicated(list(MECHANISM_KEY_COLUMNS)).sum()
    )
    latent_duplicates = int(latent.duplicated(list(LATENT_KEY_COLUMNS)).sum())
    distortion_columns = (
        "ecological_overlap_tv",
        "finite_record_tv",
        "total_composition_tv",
    )
    finite_diagnostics = bool(
        all(column in diagnostics for column in distortion_columns)
        and np.isfinite(
            diagnostics.loc[:, distortion_columns].to_numpy(dtype=float)
        ).all()
        and diagnostics.loc[:, distortion_columns]
        .apply(lambda column: column.between(0.0, 1.0))
        .all()
        .all()
    )
    finite_latent = bool(
        all(column in latent for column in PRIMARY_METRICS)
        and np.isfinite(latent.loc[:, PRIMARY_METRICS].to_numpy(dtype=float)).all()
    )
    stable_latent = bool(
        all(column in latent for column in STABILITY_COLUMNS)
        and np.isfinite(latent.loc[:, STABILITY_COLUMNS].to_numpy(dtype=float)).all()
        and latent.max_cell_mass.le(0.10).all()
        and latent.effective_cell_count.ge(50.0).all()
        and latent.lower_clip_fraction.between(0.0, 1.0).all()
        and "solver_converged" in latent
        and latent.solver_converged.eq(True).all()
    )
    labels_valid = bool(
        "feature_basis" in latent
        and latent.feature_basis.eq("linear").all()
        and "model_regularization" in latent
        and latent.model_regularization.eq(PRIMARY_REGULARIZATION).all()
        and latent.background_arm.eq(LATENT_ARM).all()
        and "evaluation_hash" in latent
        and latent.evaluation_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    )
    landscape_hashes = sorted(
        set(diagnostics.landscape_hash.astype(str)).union(
            latent.landscape_hash.astype(str)
        )
    ) if "landscape_hash" in diagnostics and "landscape_hash" in latent else []
    one_landscape = len(landscape_hashes) == 1
    failure_path = destination / "mechanism_failures.csv"
    failures = len(pd.read_csv(failure_path)) if failure_path.exists() else 0
    passed = bool(
        missing_diagnostics.empty
        and unexpected_diagnostics.empty
        and missing_latent.empty
        and unexpected_latent.empty
        and diagnostics_duplicates == 0
        and latent_duplicates == 0
        and failures == 0
        and finite_diagnostics
        and finite_latent
        and stable_latent
        and labels_valid
        and one_landscape
    )
    return {
        "status": "passed" if passed else "failed",
        "expected_diagnostics": int(len(expected_diagnostics)),
        "completed_diagnostics": int(len(diagnostics)),
        "missing_diagnostics": int(len(missing_diagnostics)),
        "unexpected_diagnostics": int(len(unexpected_diagnostics)),
        "duplicate_diagnostics": diagnostics_duplicates,
        "expected_latent": int(len(expected_latent)),
        "completed_latent": int(len(latent)),
        "missing_latent": int(len(missing_latent)),
        "unexpected_latent": int(len(unexpected_latent)),
        "duplicate_latent": latent_duplicates,
        "failures": int(failures),
        "finite_diagnostics": finite_diagnostics,
        "finite_latent_metrics": finite_latent,
        "stable_latent_predictions": stable_latent,
        "labels_valid": labels_valid,
        "landscape_hashes": landscape_hashes,
        "missing_diagnostic_keys": missing_diagnostics.to_dict(orient="records"),
        "unexpected_diagnostic_keys": unexpected_diagnostics.to_dict(orient="records"),
        "missing_latent_keys": missing_latent.to_dict(orient="records"),
        "unexpected_latent_keys": unexpected_latent.to_dict(orient="records"),
        "configuration_hash": hashlib.sha256(
            repr(config).encode("utf-8")
        ).hexdigest(),
    }
