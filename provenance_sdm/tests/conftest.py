from pathlib import Path

import pytest

from provenance_sdm.config import StudyConfig, load_study_config


@pytest.fixture
def study_config() -> StudyConfig:
    path = Path(__file__).parents[1] / "config" / "study.yaml"
    return load_study_config(path)
