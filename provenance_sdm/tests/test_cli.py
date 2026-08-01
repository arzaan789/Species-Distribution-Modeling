from __future__ import annotations

import pytest

from provenance_sdm.cli import build_parser, main


def test_help_exposes_simulation_and_audit_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "simulate" in output
    assert "audit-simulation" in output
    assert "summarize-simulation" in output
    assert "figures-simulation" in output
    assert "gbif-resolve" in output
    assert "gbif-request" in output
    assert "gbif-status" in output
    assert "gbif-retrieve" in output
    assert "build-grid" in output
    assert "clean-gbif" in output
    assert "run-empirical" in output
    assert "figures-empirical" in output
    assert "deepmaxent-pilot" in output
    assert "deepmaxent-gate" in output
    assert "run-deepmaxent" in output
    assert "export-manuscript" in output
    assert "audit-all" in output


def test_clean_gbif_requires_the_saved_request_manifest(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "clean-gbif",
                "--config",
                "config.yaml",
                "--archive",
                "download.zip",
                "--grid",
                "grid.parquet",
                "--taxa",
                "taxa.json",
                "--output",
                "outputs",
            ]
        )

    assert exc.value.code == 2
    assert "--request" in capsys.readouterr().err
