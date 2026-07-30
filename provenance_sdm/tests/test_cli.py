from __future__ import annotations

import pytest

from provenance_sdm.cli import main


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
