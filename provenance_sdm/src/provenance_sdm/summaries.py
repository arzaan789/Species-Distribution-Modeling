"""Paired contrasts and hierarchy-preserving uncertainty summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from provenance_sdm.simulation_runner import PRIMARY_METRICS


PAIR_KEYS = ("community_seed", "alignment", "bias_level", "species_id")
PRIMARY_ARMS = ("conventional_tgb", "pm_tgb")


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return per-species PM-TGB minus conventional-TGB effects."""

    required = {*PAIR_KEYS, "background_arm", *PRIMARY_METRICS}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"simulation metrics are missing columns: {sorted(missing)}")
    primary = metrics.query("background_arm in @PRIMARY_ARMS").copy()
    counts = primary.groupby(list(PAIR_KEYS)).background_arm.agg(
        ["size", "nunique"]
    )
    if not counts["size"].eq(2).all() or not counts["nunique"].eq(2).all():
        raise ValueError("primary contrasts require complete unique arm pairs")

    conventional = primary.query(
        "background_arm == 'conventional_tgb'"
    ).set_index(list(PAIR_KEYS))
    provenance = primary.query("background_arm == 'pm_tgb'").set_index(
        list(PAIR_KEYS)
    )
    if not conventional.index.equals(provenance.index):
        conventional = conventional.sort_index()
        provenance = provenance.sort_index()
    if not conventional.index.equals(provenance.index):
        raise ValueError("primary contrasts require complete aligned pairs")

    metadata_columns = [
        column
        for column in (
            "record_count",
            "niche_breadth",
            "source_distribution_distance",
        )
        if column in provenance
    ]
    frames = []
    for metric in PRIMARY_METRICS:
        frame = provenance.loc[:, metadata_columns].copy()
        frame["unsupported_mass"] = (
            provenance["unsupported_mass"] if "unsupported_mass" in provenance else 0.0
        )
        frame["metric"] = metric
        frame["conventional_value"] = conventional[metric]
        frame["pm_tgb_value"] = provenance[metric]
        frame["effect"] = frame.pm_tgb_value - frame.conventional_value
        frame["contrast"] = "pm_tgb_minus_conventional_tgb"
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)


def hierarchical_bootstrap(
    metrics: pd.DataFrame,
    n_boot: int = 2_000,
    seed: int = 20260730,
) -> pd.DataFrame:
    """Bootstrap communities, then species, while retaining paired scenarios."""

    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    effects = paired_effects(metrics)
    summary_keys = ["metric", "alignment", "bias_level"]
    point = effects.groupby(summary_keys).effect.agg(["mean", "size"]).reset_index()
    point = point.rename(columns={"mean": "estimate", "size": "n_pairs"})
    generator = np.random.default_rng(seed)
    communities = effects.community_seed.unique()
    bootstrap_rows = []

    for bootstrap_id in range(n_boot):
        sampled_tables = []
        for community_draw, community in enumerate(
            generator.choice(communities, size=len(communities), replace=True)
        ):
            community_rows = effects.query("community_seed == @community")
            species = community_rows.species_id.unique()
            for species_draw, species_id in enumerate(
                generator.choice(species, size=len(species), replace=True)
            ):
                selected = community_rows.query("species_id == @species_id").copy()
                selected["_community_draw"] = community_draw
                selected["_species_draw"] = species_draw
                sampled_tables.append(selected)
        sampled = pd.concat(sampled_tables, ignore_index=True)
        estimate = sampled.groupby(summary_keys).effect.mean().reset_index()
        estimate["bootstrap_id"] = bootstrap_id
        bootstrap_rows.append(estimate)

    draws = pd.concat(bootstrap_rows, ignore_index=True)
    intervals = (
        draws.groupby(summary_keys)
        .effect.quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "lower", 0.975: "upper"})
        .reset_index()
    )
    output = point.merge(intervals, on=summary_keys, validate="one_to_one")
    output["contrast"] = "pm_tgb_minus_conventional_tgb"
    return output.sort_values(summary_keys).reset_index(drop=True)
