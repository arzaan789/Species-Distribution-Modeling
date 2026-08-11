"""Paired contrasts and hierarchy-preserving uncertainty summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

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
    pair_columns = ["community_seed", "species_id"]
    pair_rows = effects.loc[:, pair_columns].drop_duplicates()
    pair_index = pd.MultiIndex.from_frame(pair_rows)
    matrix = effects.pivot(
        index=pair_columns,
        columns=summary_keys,
        values="effect",
    ).reindex(pair_index)
    sampler = _HierarchySampler(pair_rows)
    values = matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    values = np.nan_to_num(values)
    draws = np.empty((n_boot, values.shape[1]), dtype=float)
    for draw in range(n_boot):
        positions = sampler.sample_positions(generator)
        weights = np.bincount(positions, minlength=len(pair_rows))
        denominators = weights @ finite
        draws[draw] = np.divide(
            weights @ values,
            denominators,
            out=np.full(values.shape[1], np.nan),
            where=denominators > 0,
        )
    intervals = matrix.columns.to_frame(index=False)
    intervals["lower"] = np.nanquantile(draws, 0.025, axis=0)
    intervals["upper"] = np.nanquantile(draws, 0.975, axis=0)
    output = point.merge(intervals, on=summary_keys, validate="one_to_one")
    output["contrast"] = "pm_tgb_minus_conventional_tgb"
    return output.sort_values(summary_keys).reset_index(drop=True)


def _orientation(metric: str) -> int:
    return -1 if metric in {"integrated_error", "response_curve_error"} else 1


def oriented_paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return primary paired effects oriented so positive means improvement."""

    effects = paired_effects(metrics)
    effects["orientation"] = effects.metric.map(_orientation).astype(int)
    effects["oriented_effect"] = effects.effect * effects.orientation
    return effects


def diagnostic_arm_effects(
    primary_metrics: pd.DataFrame,
    latent_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Align latent-mixture results with observed and conventional TGB arms."""

    required_primary = {*PAIR_KEYS, "background_arm", *PRIMARY_METRICS}
    required_latent = {*PAIR_KEYS, "background_arm", *PRIMARY_METRICS}
    if missing := required_primary.difference(primary_metrics.columns):
        raise ValueError(f"primary metrics are missing columns: {sorted(missing)}")
    if missing := required_latent.difference(latent_metrics.columns):
        raise ValueError(f"latent metrics are missing columns: {sorted(missing)}")
    latent = latent_metrics.query(
        "background_arm == 'latent_mixture_tgb'"
    ).set_index(list(PAIR_KEYS)).sort_index()
    if latent.empty or not latent.index.is_unique:
        raise ValueError("latent metrics require one unique row per pair")
    frames = []
    for comparator_arm in ("conventional_tgb", "pm_tgb"):
        comparator = primary_metrics.query(
            "background_arm == @comparator_arm"
        ).set_index(list(PAIR_KEYS)).sort_index()
        if not comparator.index.is_unique or not comparator.index.equals(latent.index):
            raise ValueError(
                f"latent and {comparator_arm} metrics require complete unique aligned pairs"
            )
        for metric in PRIMARY_METRICS:
            frame = pd.DataFrame(index=latent.index)
            frame["metric"] = metric
            frame["comparator_value"] = comparator[metric]
            frame["latent_mixture_value"] = latent[metric]
            frame["effect"] = latent[metric] - comparator[metric]
            frame["orientation"] = _orientation(metric)
            frame["oriented_effect"] = frame.effect * frame.orientation
            frame["contrast"] = (
                f"latent_mixture_tgb_minus_{comparator_arm}"
            )
            frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)


class _HierarchySampler:
    """Pre-index a community-species hierarchy for repeated bootstrap draws."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self._communities = rows.community_seed.unique()
        community_values = rows.community_seed.to_numpy()
        species_values = rows.species_id.to_numpy()
        self._positions: dict[object, np.ndarray] = {}
        for community in self._communities:
            positions = np.flatnonzero(community_values == community)
            if pd.Index(species_values[positions]).has_duplicates:
                raise ValueError(
                    "hierarchical samples require one row per community-species pair"
                )
            self._positions[community] = positions

    def sample_positions(
        self,
        generator: np.random.Generator,
    ) -> np.ndarray:
        sampled = []
        communities = generator.choice(
            self._communities,
            size=len(self._communities),
            replace=True,
        )
        for community in communities:
            positions = self._positions[community]
            sampled.append(
                generator.choice(positions, size=len(positions), replace=True)
            )
        return np.concatenate(sampled)


def hierarchical_effect_bootstrap(
    effects: pd.DataFrame,
    n_boot: int = 2_000,
    seed: int = 20260730,
) -> pd.DataFrame:
    """Summarize oriented effects by resampling communities then species."""

    required = {
        *PAIR_KEYS,
        "metric",
        "contrast",
        "oriented_effect",
    }
    if missing := required.difference(effects.columns):
        raise ValueError(f"effect rows are missing columns: {sorted(missing)}")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    unique_keys = [*PAIR_KEYS, "metric", "contrast"]
    if effects.duplicated(unique_keys).any():
        raise ValueError("effect rows must have unique pair-metric-contrast keys")
    group_keys = ["metric", "alignment", "bias_level", "contrast"]
    generator = np.random.default_rng(seed)
    output = []
    for key, rows in effects.groupby(group_keys, sort=True):
        sampler = _HierarchySampler(rows)
        values = rows.oriented_effect.to_numpy(dtype=float)
        draws = np.empty(n_boot, dtype=float)
        for draw in range(n_boot):
            positions = sampler.sample_positions(generator)
            draws[draw] = values[positions].mean()
        output.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "estimate": float(rows.oriented_effect.mean()),
                "lower": float(np.quantile(draws, 0.025)),
                "upper": float(np.quantile(draws, 0.975)),
                "n_pairs": int(len(rows)),
                "bootstrap_draws": int(n_boot),
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(output).sort_values(group_keys).reset_index(drop=True)


def _rank_correlation(rows: pd.DataFrame) -> float:
    left = rows.ecological_overlap_tv.to_numpy(dtype=float)
    right = rows.oriented_effect.to_numpy(dtype=float)
    if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return np.nan
    return float(scipy.stats.spearmanr(left, right).statistic)


def mechanism_correlations(
    primary_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    n_boot: int = 2_000,
    seed: int = 20260730,
) -> pd.DataFrame:
    """Relate known ecological source distortion to oriented PM-TGB effects."""

    diagnostic_required = {*PAIR_KEYS, "ecological_overlap_tv"}
    if missing := diagnostic_required.difference(diagnostics.columns):
        raise ValueError(f"mechanism diagnostics are missing columns: {sorted(missing)}")
    if diagnostics.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("mechanism diagnostics must have unique pair keys")
    effects = oriented_paired_effects(primary_metrics)
    effect_pairs = effects.loc[:, PAIR_KEYS].drop_duplicates()
    diagnostic_pairs = diagnostics.loc[:, PAIR_KEYS]
    left_only = effect_pairs.merge(
        diagnostic_pairs,
        on=list(PAIR_KEYS),
        how="left",
        indicator=True,
    )._merge.ne("both").any()
    right_only = diagnostic_pairs.merge(
        effect_pairs,
        on=list(PAIR_KEYS),
        how="left",
        indicator=True,
    )._merge.ne("both").any()
    if left_only or right_only:
        raise ValueError("mechanism diagnostics must completely match primary pairs")
    merged = effects.merge(
        diagnostics.loc[:, [*PAIR_KEYS, "ecological_overlap_tv"]],
        on=list(PAIR_KEYS),
        how="left",
        validate="many_to_one",
    )
    if (
        not np.isfinite(merged.ecological_overlap_tv.to_numpy(dtype=float)).all()
        or not merged.ecological_overlap_tv.between(0.0, 1.0).all()
    ):
        raise ValueError("ecological overlap distortion must be finite and bounded")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    group_keys = ["metric", "alignment", "bias_level"]
    generator = np.random.default_rng(seed)
    output = []
    for key, rows in merged.groupby(group_keys, sort=True):
        sampler = _HierarchySampler(rows)
        left = rows.ecological_overlap_tv.to_numpy(dtype=float)
        right = rows.oriented_effect.to_numpy(dtype=float)
        draws = np.empty(n_boot, dtype=float)
        for draw in range(n_boot):
            positions = sampler.sample_positions(generator)
            sampled = pd.DataFrame(
                {
                    "ecological_overlap_tv": left[positions],
                    "oriented_effect": right[positions],
                }
            )
            draws[draw] = _rank_correlation(sampled)
        finite_draws = draws[np.isfinite(draws)]
        lower = float(np.quantile(finite_draws, 0.025)) if finite_draws.size else np.nan
        upper = float(np.quantile(finite_draws, 0.975)) if finite_draws.size else np.nan
        output.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "estimate": _rank_correlation(rows),
                "lower": lower,
                "upper": upper,
                "n_pairs": int(len(rows)),
                "bootstrap_draws": int(n_boot),
                "bootstrap_seed": int(seed),
                "finite_bootstrap_draws": int(finite_draws.size),
                "method": "Spearman correlation with community-species bootstrap",
            }
        )
    return pd.DataFrame(output).sort_values(group_keys).reset_index(drop=True)
