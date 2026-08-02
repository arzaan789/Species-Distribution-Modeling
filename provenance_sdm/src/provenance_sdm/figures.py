"""Neutral, reproducible figures generated from frozen tidy artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance_sdm.summaries import (
    diagnostic_arm_effects,
    hierarchical_effect_bootstrap,
    oriented_paired_effects,
    paired_effects,
)


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _workflow_figure(path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10, 3))
    axis.axis("off")
    labels = (
        "Ecological\nsuitability",
        "Recording-programme\neffort",
        "Latent programme\nallocation",
        "Presence-only\nrecords",
        "Observed source\ncomposition",
        "Background\narms",
        "Truth-based\nevaluation",
    )
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for x_position, label in zip(x_positions, labels, strict=True):
        axis.text(
            x_position,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#e8eef5"},
            transform=axis.transAxes,
        )
    for left, right in zip(x_positions[:-1], x_positions[1:], strict=True):
        axis.annotate(
            "",
            xy=(right - 0.07, 0.5),
            xytext=(left + 0.07, 0.5),
            arrowprops={"arrowstyle": "->", "color": "#4a5568"},
            xycoords=axis.transAxes,
        )
    axis.set_title("Simulation and evaluation workflow")
    return _save(figure, path)


def _effect_figure(effects: pd.DataFrame, path: Path) -> Path:
    selected = effects.query("metric == 'suitability_spearman'")
    summary = (
        selected.groupby(["alignment", "bias_level"]).effect.agg(["mean", "sem"])
        .reset_index()
    )
    alignments = [item for item in ("low", "partial", "high") if item in set(summary.alignment)]
    figure, axis = plt.subplots(figsize=(7, 4))
    for bias_index, (bias_level, rows) in enumerate(summary.groupby("bias_level")):
        rows = rows.set_index("alignment").reindex(alignments)
        offset = (bias_index - (summary.bias_level.nunique() - 1) / 2) * 0.12
        axis.errorbar(
            np.arange(len(alignments)) + offset,
            rows["mean"],
            yerr=1.96 * rows["sem"].fillna(0),
            marker="o",
            capsize=3,
            label=bias_level,
        )
    axis.axhline(0, color="#4a5568", linewidth=1)
    axis.set_xticks(range(len(alignments)), alignments)
    axis.set_xlabel("Taxonomy–programme alignment")
    axis.set_ylabel("PM-TGB − conventional TGB\nsuitability Spearman")
    axis.set_title("Paired truth-recovery contrast by observation scenario")
    axis.legend(title="Bias intensity")
    return _save(figure, path)


def _condition_figure(effects: pd.DataFrame, path: Path) -> Path:
    selected = effects.query("metric == 'suitability_spearman'")
    variables = (
        ("source_distribution_distance", "Source-distribution distance"),
        ("record_count", "Focal record count"),
        ("niche_breadth", "Niche breadth"),
        ("unsupported_mass", "Unsupported source mass"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)
    for axis, (column, label) in zip(axes.ravel(), variables, strict=True):
        axis.scatter(selected[column], selected.effect, alpha=0.6, s=18)
        axis.axhline(0, color="#718096", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("Paired Spearman contrast")
    figure.suptitle("Observed contrast across predeclared study conditions")
    return _save(figure, path)


def write_simulation_figures(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write the two frozen primary simulation figure panels."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    effects = paired_effects(metrics)
    return (
        _workflow_figure(destination / "simulation_workflow.png"),
        _effect_figure(effects, destination / "paired_truth_contrasts.png"),
    )


def _source_composition_figure(
    primary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> Path:
    effects = oriented_paired_effects(primary).query(
        "metric == 'suitability_spearman'"
    )
    keys = ["community_seed", "alignment", "bias_level", "species_id"]
    selected = effects.merge(
        diagnostics.loc[:, [*keys, "ecological_overlap_tv"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if selected.ecological_overlap_tv.isna().any():
        raise ValueError("mechanism diagnostics are incomplete for plotted pairs")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for (alignment, bias_level), rows in selected.groupby(
        ["alignment", "bias_level"], sort=True
    ):
        axis.scatter(
            rows.ecological_overlap_tv,
            rows.oriented_effect,
            alpha=0.45,
            s=16,
            label=f"{alignment}, {bias_level}",
        )
    axis.axhline(0, color="#4a5568", linewidth=0.9)
    axis.set_xlabel("Ecological-overlap distortion (total variation)")
    axis.set_ylabel("Oriented PM-TGB contrast\n(positive = better truth recovery)")
    axis.set_title("Observed source composition mixes allocation and ecology")
    axis.legend(fontsize="x-small", ncol=2)
    return _save(figure, path)


def _latent_contrast_figure(
    primary: pd.DataFrame,
    latent: pd.DataFrame,
    path: Path,
    *,
    n_boot: int,
    seed: int,
) -> Path:
    effects = diagnostic_arm_effects(primary, latent)
    summary = hierarchical_effect_bootstrap(
        effects,
        n_boot=n_boot,
        seed=seed,
    ).query("metric == 'suitability_spearman'")
    scenarios = sorted(
        {(row.alignment, row.bias_level) for row in summary.itertuples()}
    )
    labels = [f"{alignment}\n{bias}" for alignment, bias in scenarios]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    contrasts = tuple(summary.contrast.unique())
    for contrast_index, contrast in enumerate(contrasts):
        rows = summary.query("contrast == @contrast").set_index(
            ["alignment", "bias_level"]
        ).reindex(scenarios)
        offset = (contrast_index - (len(contrasts) - 1) / 2) * 0.16
        axis.errorbar(
            np.arange(len(scenarios)) + offset,
            rows.estimate,
            yerr=np.vstack(
                [rows.estimate - rows.lower, rows.upper - rows.estimate]
            ),
            marker="o",
            capsize=3,
            label=contrast.replace("_", " "),
        )
    axis.axhline(0, color="#4a5568", linewidth=0.9)
    axis.set_xticks(range(len(scenarios)), labels)
    axis.set_xlabel("Alignment and bias scenario")
    axis.set_ylabel("Oriented latent-mixture contrast")
    axis.set_title("Diagnostic access to latent allocation weights")
    axis.legend(fontsize="small")
    return _save(figure, path)


def write_mechanism_figures(
    primary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    latent: pd.DataFrame,
    output_dir: Path,
    *,
    n_boot: int = 2_000,
    seed: int = 20260730,
) -> tuple[Path, ...]:
    """Write mechanism and latent diagnostic panels from tidy artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return (
        _source_composition_figure(
            primary,
            diagnostics,
            destination / "source_composition_mechanism.png",
        ),
        _latent_contrast_figure(
            primary,
            latent,
            destination / "latent_mixture_contrasts.png",
            n_boot=n_boot,
            seed=seed,
        ),
    )


def _source_overlap_figure(metrics: pd.DataFrame, path: Path) -> Path:
    primary = metrics[
        metrics.provenance_level.eq("dataset")
        & metrics.block_width_m.eq(50_000)
        & metrics.background_arm.isin(("conventional_tgb", "pm_tgb"))
    ]
    paired = primary.pivot(
        index=["species", "fold_id", "source_distance"],
        columns="background_arm",
        values="auc",
    ).dropna()
    paired["auc_contrast"] = paired.pm_tgb - paired.conventional_tgb
    paired = paired.reset_index()
    figure, axis = plt.subplots(figsize=(7, 4))
    for species, rows in paired.groupby("species"):
        axis.scatter(
            rows.source_distance,
            rows.auc_contrast,
            label=species.replace("_", " "),
            alpha=0.75,
        )
    axis.axhline(0, color="#718096", linewidth=0.9)
    axis.set_xlabel("Focal–target source-distribution distance")
    axis.set_ylabel("PM-TGB − conventional TGB AUC")
    axis.set_title("Held-out discrimination contrast and source mismatch")
    axis.legend(title="Species", fontsize="small")
    return _save(figure, path)


def empirical_map_difference(maps: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic PM-minus-conventional map per million cells."""

    required = {
        "cell_id",
        "x",
        "y",
        "species",
        "background_arm",
        "predicted_suitability",
    }
    if missing := required.difference(maps.columns):
        raise ValueError(f"empirical maps are missing columns: {sorted(missing)}")
    available = sorted(maps.species.dropna().astype(str).unique())
    if not available:
        raise ValueError("empirical maps contain no species")
    species = available[0]
    selected = maps[
        maps.species.eq(species)
        & maps.background_arm.isin(("conventional_tgb", "pm_tgb"))
    ]
    wide = selected.pivot(
        index=["cell_id", "x", "y"],
        columns="background_arm",
        values="predicted_suitability",
    ).dropna()
    if not {"conventional_tgb", "pm_tgb"} <= set(wide.columns):
        raise ValueError("empirical map contrast requires both TGB arms")
    difference = wide.pm_tgb - wide.conventional_tgb
    result = wide.index.to_frame(index=False)
    result["species"] = species
    result["difference_per_million"] = difference.to_numpy(dtype=float) * 1_000_000
    return result


def _map_disagreement_figure(maps: pd.DataFrame, path: Path) -> Path:
    difference = empirical_map_difference(maps)
    species = str(difference.species.iloc[0])
    values = difference.difference_per_million
    scale = float(np.abs(values).max())
    if scale == 0:
        scale = 1.0
    figure, axis = plt.subplots(figsize=(6, 5))
    points = axis.scatter(
        difference.x,
        difference.y,
        c=values,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        s=12,
    )
    figure.colorbar(
        points,
        ax=axis,
        label="PM-TGB − conventional mass per million cells",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("Projected x")
    axis.set_ylabel("Projected y")
    axis.set_title(f"Background-arm map contrast: {species.replace('_', ' ')}")
    return _save(figure, path)


def write_empirical_figures(
    metrics: pd.DataFrame,
    maps: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write neutral primary empirical source and map comparison panels."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return (
        _source_overlap_figure(
            metrics,
            destination / "empirical_source_contrasts.png",
        ),
        _map_disagreement_figure(
            maps,
            destination / "empirical_map_contrast.png",
        ),
    )
