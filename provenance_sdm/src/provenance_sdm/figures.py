"""Neutral, reproducible figures generated from frozen tidy artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance_sdm.summaries import paired_effects


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
        "Presence-only\nrecords",
        "Four background\narms",
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
    """Write the three predeclared simulation figure panels."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    effects = paired_effects(metrics)
    return (
        _workflow_figure(destination / "simulation_workflow.png"),
        _effect_figure(effects, destination / "paired_truth_contrasts.png"),
        _condition_figure(effects, destination / "contrast_conditions.png"),
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


def _map_disagreement_figure(maps: pd.DataFrame, path: Path) -> Path:
    species = str(maps.species.iloc[0])
    selected = maps[
        maps.species.eq(species)
        & maps.background_arm.isin(("conventional_tgb", "pm_tgb"))
    ]
    wide = selected.pivot(
        index=["cell_id", "x", "y"],
        columns="background_arm",
        values="predicted_suitability",
    ).dropna()
    difference = wide.pm_tgb - wide.conventional_tgb
    scale = float(np.abs(difference).max())
    if scale == 0:
        scale = 1.0
    figure, axis = plt.subplots(figsize=(6, 5))
    points = axis.scatter(
        wide.index.get_level_values("x"),
        wide.index.get_level_values("y"),
        c=difference,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        s=12,
    )
    figure.colorbar(points, ax=axis, label="PM-TGB − conventional suitability")
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
