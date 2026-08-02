"""Known-process diagnostics for simulated recording-source composition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from provenance_sdm.observation import ObservedCommunity
from provenance_sdm.virtual_species import SpeciesTruth


def _truth_by_id(observed: ObservedCommunity) -> dict[str, SpeciesTruth]:
    truth = {species.species_id: species for species in observed.truth}
    if len(truth) != len(observed.truth):
        raise ValueError("simulation truth contains duplicate species identifiers")
    return truth


def _focal_truth(
    observed: ObservedCommunity,
    focal_species: str,
) -> SpeciesTruth:
    truth = _truth_by_id(observed)
    if focal_species not in truth:
        raise ValueError(f"unknown focal species: {focal_species!r}")
    return truth[focal_species]


def _latent_source_mass(
    observed: ObservedCommunity,
    focal_species: str,
) -> pd.Series:
    programme_ids = pd.Index(
        [programme.programme_id for programme in observed.programme_effort],
        name="programme_id",
    )
    if programme_ids.empty or not programme_ids.is_unique:
        raise ValueError("recording programmes must be non-empty and unique")
    rows = observed.source_mixtures.query("species_id == @focal_species")
    if rows.programme_id.duplicated().any():
        raise ValueError("focal source mixture contains duplicate programmes")
    latent = rows.set_index("programme_id").weight.reindex(programme_ids)
    _validate_probability_mass(latent, "latent source mass")
    return latent.astype(float)


def _validate_probability_mass(mass: pd.Series, diagnostic: str) -> None:
    values = mass.to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{diagnostic} must be non-empty and finite")
    if np.any(values < 0):
        raise ValueError(f"{diagnostic} must be non-negative")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError(f"{diagnostic} must sum to one")


def total_variation(left: pd.Series, right: pd.Series) -> float:
    """Return total-variation distance between two labelled source masses."""

    _validate_probability_mass(left, "left source mass")
    _validate_probability_mass(right, "right source mass")
    labels = left.index.union(right.index, sort=False)
    left_mass = left.reindex(labels, fill_value=0.0).astype(float)
    right_mass = right.reindex(labels, fill_value=0.0).astype(float)
    return float(0.5 * np.abs(left_mass - right_mass).sum())


def expected_source_composition(
    observed: ObservedCommunity,
    focal_species: str,
) -> pd.Series:
    """Return expected observed source mass after ecological overlap."""

    species = _focal_truth(observed, focal_species)
    suitability = np.asarray(species.suitability, dtype=float)
    if (
        suitability.size == 0
        or not np.isfinite(suitability).all()
        or np.any(suitability < 0)
        or suitability.sum() <= 0
    ):
        raise ValueError("focal suitability must be finite and non-negative")
    latent = _latent_source_mass(observed, focal_species)
    overlap_rows = []
    for programme in observed.programme_effort:
        intensity = np.asarray(programme.intensity, dtype=float)
        if intensity.shape != suitability.shape or not np.isfinite(intensity).all():
            raise ValueError("programme effort must match finite focal suitability")
        if np.any(intensity < 0) or intensity.sum() <= 0:
            raise ValueError("programme effort must be non-negative and non-empty")
        overlap_rows.append(
            (programme.programme_id, float(intensity @ suitability))
        )
    overlap = pd.Series(dict(overlap_rows), dtype=float).reindex(latent.index)
    expected = latent * overlap
    total = float(expected.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("expected source composition must have positive mass")
    expected /= total
    return expected


def mechanism_row(
    observed: ObservedCommunity,
    focal_species: str,
) -> dict[str, object]:
    """Return the preregistered distortion decomposition for one focal species."""

    species = _focal_truth(observed, focal_species)
    latent = _latent_source_mass(observed, focal_species)
    expected = expected_source_composition(observed, focal_species)
    records = observed.records.query("species_id == @focal_species")
    if records.empty or records.programme_id.isna().any():
        raise ValueError("focal species must have complete observed source labels")
    unknown_sources = set(records.programme_id).difference(latent.index)
    if unknown_sources:
        raise ValueError(
            f"focal records contain unknown programmes: {sorted(unknown_sources)}"
        )
    realized = (
        records.programme_id.value_counts(normalize=True, sort=False)
        .reindex(latent.index, fill_value=0.0)
        .astype(float)
    )
    return {
        "record_count": int(len(records)),
        "niche_breadth": float(species.niche_breadth),
        "ecological_overlap_tv": total_variation(expected, latent),
        "finite_record_tv": total_variation(realized, expected),
        "total_composition_tv": total_variation(realized, latent),
    }
