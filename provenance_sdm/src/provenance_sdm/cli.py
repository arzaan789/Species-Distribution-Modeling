"""Command-line entry points for the reproducible analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from provenance_sdm.config import load_study_config
from provenance_sdm.landscape import Landscape
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_study_config(arguments.config)
    if arguments.command == "simulate":
        landscape = _load_landscape(arguments.landscape, arguments.crs)
        run_simulation(config, landscape, config.output_dir)
        return 0
    report = audit_simulation(arguments.results, config)
    return 0 if report["status"] == "passed" else 1
