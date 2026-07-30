"""Provenance-matched target-group background weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProvenanceWeights:
    weights: pd.Series
    focal_mass: pd.Series
    supported_mass: float
    unsupported_mass: float


def pm_tgb_weights(
    focal_sources: pd.Series,
    candidate_sources: pd.Series,
) -> ProvenanceWeights:
    """Match candidate source mass to the focal source distribution.

    Focal mass from sources without candidates is distributed over the full
    pooled target group, which is the conventional TGB fallback.
    """

    if focal_sources.empty or candidate_sources.empty:
        raise ValueError("source series must be non-empty")
    if focal_sources.isna().any() or candidate_sources.isna().any():
        raise ValueError("source labels must be complete")
    if not candidate_sources.index.is_unique:
        raise ValueError("candidate indices must be unique")

    focal_mass = focal_sources.value_counts(normalize=True, sort=False)
    candidate_counts = candidate_sources.value_counts(sort=False)
    supported_sources = focal_mass.index.intersection(
        candidate_counts.index,
        sort=False,
    )

    weights = pd.Series(0.0, index=candidate_sources.index, dtype=float)
    for source in supported_sources:
        source_rows = candidate_sources.eq(source)
        weights.loc[source_rows] = float(focal_mass.loc[source]) / int(
            source_rows.sum()
        )

    unsupported_sources = focal_mass.index.difference(
        supported_sources,
        sort=False,
    )
    unsupported_mass = float(focal_mass.loc[unsupported_sources].sum())
    weights += unsupported_mass / len(candidate_sources)

    total = float(weights.sum())
    if not np.isfinite(weights.to_numpy()).all() or not np.isfinite(total):
        raise ValueError("calculated provenance weights must be finite")
    if total <= 0 or weights.lt(0).any():
        raise ValueError("calculated provenance weights must be non-negative")
    weights /= total

    return ProvenanceWeights(
        weights=weights,
        focal_mass=focal_mass,
        supported_mass=1.0 - unsupported_mass,
        unsupported_mass=unsupported_mass,
    )
