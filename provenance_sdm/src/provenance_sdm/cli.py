"""Command-line entry points for the reproducible analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from provenance_sdm.config import load_study_config
from provenance_sdm.empirical import (
    EmpiricalInputs,
    attach_nearest_grid_cells,
    clean_occurrences,
    read_gbif_archive,
    run_empirical,
)
from provenance_sdm.gbif import GBIFClient, TaxonMatch
from provenance_sdm.landscape import Landscape, landscape_from_geographic_frame
from provenance_sdm.manifests import sha256_file, write_manifest
from provenance_sdm.simulation_runner import audit_simulation, run_simulation


def _load_landscape(path: Path, crs: str) -> Landscape:
    cells = pd.read_parquet(path)
    fixed = {"cell_id", "x", "y", "area_weight"}
    feature_names = tuple(column for column in cells if column not in fixed)
    return Landscape(
        cells=cells,
        feature_names=feature_names,
        crs=crs,
        feature_means={name: 0.0 for name in feature_names},
        feature_scales={name: 1.0 for name in feature_names},
    )


def _load_taxon_keys(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return {
            str(item["requested_name"]): int(item["taxon_key"])
            for item in payload["taxa"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("taxon manifest has an invalid schema") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="provenance-sdm")
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--landscape", type=Path, required=True)
    simulate.add_argument("--crs", default="EPSG:27700")
    audit = commands.add_parser("audit-simulation")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--results", type=Path, required=True)
    summarize = commands.add_parser("summarize-simulation")
    summarize.add_argument("--results", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    figures = commands.add_parser("figures-simulation")
    figures.add_argument("--results", type=Path, required=True)
    figures.add_argument("--output", type=Path, required=True)
    resolve = commands.add_parser("gbif-resolve")
    resolve.add_argument("--config", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    request = commands.add_parser("gbif-request")
    request.add_argument("--taxa", type=Path, required=True)
    request.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("gbif-status")
    status.add_argument("--download-key", required=True)
    status.add_argument("--output", type=Path, required=True)
    retrieve = commands.add_parser("gbif-retrieve")
    retrieve.add_argument("--status", type=Path, required=True)
    retrieve.add_argument("--archive", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    grid = commands.add_parser("build-grid")
    grid.add_argument("--predictors", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument(
        "--features",
        nargs="+",
        default=[
            "BSI",
            "LST",
            "MNDWI",
            "NDBI",
            "NDSI",
            "NDVI",
            "NDWI",
            "SAVI",
            "UI",
        ],
    )
    grid.add_argument("--crs", default="EPSG:27700")
    grid.add_argument("--cell-area", type=float, default=1_000_000)
    clean = commands.add_parser("clean-gbif")
    clean.add_argument("--config", type=Path, required=True)
    clean.add_argument("--archive", type=Path, required=True)
    clean.add_argument("--grid", type=Path, required=True)
    clean.add_argument("--taxa", type=Path, required=True)
    clean.add_argument("--crs", default="EPSG:27700")
    clean.add_argument("--max-distance-m", type=float, default=1_500)
    clean.add_argument("--output", type=Path, required=True)
    empirical = commands.add_parser("run-empirical")
    empirical.add_argument("--config", type=Path, required=True)
    empirical.add_argument("--records", type=Path, required=True)
    empirical.add_argument("--grid", type=Path, required=True)
    empirical.add_argument("--taxa", type=Path, required=True)
    empirical.add_argument("--crs", default="EPSG:27700")
    empirical.add_argument(
        "--block-widths",
        type=int,
        nargs="+",
        default=[25_000, 50_000, 100_000],
    )
    empirical.add_argument("--folds", type=int, default=5)
    empirical.add_argument("--output", type=Path, required=True)
    empirical_figures = commands.add_parser("figures-empirical")
    empirical_figures.add_argument("--results", type=Path, required=True)
    empirical_figures.add_argument("--maps", type=Path, required=True)
    empirical_figures.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "simulate":
        config = load_study_config(arguments.config)
        landscape = _load_landscape(arguments.landscape, arguments.crs)
        run_simulation(config, landscape, config.output_dir)
        return 0
    if arguments.command == "audit-simulation":
        config = load_study_config(arguments.config)
        report = audit_simulation(arguments.results, config)
        return 0 if report["status"] == "passed" else 1
    if arguments.command == "gbif-resolve":
        config = load_study_config(arguments.config)
        names = list(
            dict.fromkeys(
                name
                for species in config.empirical_species
                for name in (species.scientific_name, *species.target_group)
            )
        )
        client = GBIFClient.from_environment()
        payload = {
            "taxa": [asdict(client.resolve_taxon(name)) for name in names],
        }
        write_manifest(payload, arguments.output)
        return 0
    if arguments.command == "gbif-request":
        payload = json.loads(arguments.taxa.read_text(encoding="utf-8"))
        taxa = [TaxonMatch(**item) for item in payload["taxa"]]
        submitted = GBIFClient.from_environment().submit_download(taxa)
        write_manifest(submitted, arguments.output)
        return 0
    if arguments.command == "gbif-status":
        status_payload = GBIFClient.from_environment().download_status(
            arguments.download_key
        )
        write_manifest(status_payload, arguments.output, allow_replace=True)
        return 0
    if arguments.command == "gbif-retrieve":
        completed_status = json.loads(arguments.status.read_text(encoding="utf-8"))
        archive_payload = GBIFClient.from_environment().retrieve_archive(
            completed_status,
            arguments.archive,
        )
        write_manifest(archive_payload, arguments.output)
        return 0
    if arguments.command == "build-grid":
        source = pd.read_csv(arguments.predictors)
        landscape = landscape_from_geographic_frame(
            source,
            tuple(arguments.features),
            arguments.crs,
            arguments.cell_area,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        landscape.cells.to_parquet(arguments.output, index=False)
        write_manifest(
            {
                "source": str(arguments.predictors),
                "source_sha256": sha256_file(arguments.predictors),
                "source_records": len(source),
                "retained_cells": len(landscape.cells),
                "removed_incomplete": len(source) - len(landscape.cells),
                "feature_names": list(landscape.feature_names),
                "feature_means": dict(landscape.feature_means),
                "feature_scales": dict(landscape.feature_scales),
                "crs": landscape.crs,
                "cell_area": arguments.cell_area,
            },
            arguments.output.with_suffix(".manifest.json"),
            allow_replace=True,
        )
        return 0
    if arguments.command == "clean-gbif":
        config = load_study_config(arguments.config)
        landscape = _load_landscape(arguments.grid, arguments.crs)
        taxon_keys = _load_taxon_keys(arguments.taxa)
        allowed_taxa = set(taxon_keys.values())
        attached = attach_nearest_grid_cells(
            read_gbif_archive(arguments.archive),
            landscape,
            arguments.max_distance_m,
        )
        cleaned = clean_occurrences(
            attached,
            set(landscape.cells.cell_id),
            allowed_taxa,
        )
        arguments.output.mkdir(parents=True, exist_ok=True)
        records = cleaned.records.copy()
        records.insert(0, "record_id", range(len(records)))
        records.to_parquet(
            arguments.output / "clean_occurrences.parquet",
            index=False,
        )
        cleaned.audit.to_csv(
            arguments.output / "occurrence_cleaning_audit.csv",
            index=False,
        )
        return 0
    if arguments.command == "run-empirical":
        config = load_study_config(arguments.config)
        return_path = run_empirical(
            EmpiricalInputs(
                config=config,
                records=pd.read_parquet(arguments.records),
                landscape=_load_landscape(arguments.grid, arguments.crs),
                taxon_keys=_load_taxon_keys(arguments.taxa),
                block_widths=tuple(arguments.block_widths),
                n_folds=arguments.folds,
            ),
            arguments.output,
        )
        return 0 if return_path.is_file() else 1
    if arguments.command == "figures-empirical":
        from provenance_sdm.figures import write_empirical_figures

        write_empirical_figures(
            pd.read_parquet(arguments.results),
            pd.read_parquet(arguments.maps),
            arguments.output,
        )
        return 0
    metrics = pd.read_parquet(arguments.results)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.command == "summarize-simulation":
        from provenance_sdm.summaries import hierarchical_bootstrap, paired_effects

        paired_effects(metrics).to_parquet(
            arguments.output / "paired_effects.parquet",
            index=False,
        )
        hierarchical_bootstrap(metrics).to_csv(
            arguments.output / "primary_effect_intervals.csv",
            index=False,
        )
        return 0
    from provenance_sdm.figures import write_simulation_figures

    write_simulation_figures(metrics, arguments.output)
    return 0
