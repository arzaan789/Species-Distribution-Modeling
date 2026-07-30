from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import responses

from provenance_sdm.gbif import GBIFClient, TaxonMatch, build_download_predicate


@pytest.fixture
def client() -> GBIFClient:
    return GBIFClient(
        username="test-user",
        password="test-password",
        notification_email="researcher@example.org",
    )


@responses.activate
def test_resolve_taxon_uses_current_accepted_species_contract(client) -> None:
    responses.get(
        "https://api.gbif.org/v2/species/match",
        json={
            "usage": {
                "key": "7952072",
                "name": "Lepus europaeus Pallas, 1778",
                "canonicalName": "Lepus europaeus",
                "rank": "SPECIES",
                "status": "ACCEPTED",
            },
            "diagnostics": {"matchType": "EXACT", "confidence": 99},
            "synonym": False,
        },
    )

    taxon = client.resolve_taxon("Lepus europaeus")

    assert taxon == TaxonMatch(
        requested_name="Lepus europaeus",
        taxon_key=7952072,
        accepted_name="Lepus europaeus Pallas, 1778",
        confidence=99,
    )
    assert responses.calls[0].request.params["scientificName"] == "Lepus europaeus"


@responses.activate
def test_nonaccepted_or_nonspecies_taxon_match_is_rejected(client) -> None:
    responses.get(
        "https://api.gbif.org/v2/species/match",
        json={
            "usage": {
                "key": "2436691",
                "name": "Lepus",
                "rank": "GENUS",
                "status": "ACCEPTED",
            },
            "diagnostics": {"confidence": 95},
        },
    )

    with pytest.raises(ValueError, match="accepted species"):
        client.resolve_taxon("Lepus")


def test_download_predicate_is_frozen_and_contains_only_taxon_keys() -> None:
    taxa = [
        TaxonMatch("Lepus europaeus", 7952072, "Lepus europaeus", 99),
        TaxonMatch("Lepus timidus", 2436693, "Lepus timidus", 99),
    ]

    predicate = build_download_predicate(taxa)

    assert predicate == {
        "type": "and",
        "predicates": [
            {"type": "equals", "key": "COUNTRY", "value": "GB"},
            {
                "type": "equals",
                "key": "OCCURRENCE_STATUS",
                "value": "PRESENT",
            },
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {
                "type": "equals",
                "key": "HAS_GEOSPATIAL_ISSUE",
                "value": "false",
            },
            {
                "type": "greaterThanOrEquals",
                "key": "YEAR",
                "value": "2022",
            },
            {
                "type": "lessThanOrEquals",
                "key": "YEAR",
                "value": "2025",
            },
            {
                "type": "in",
                "key": "TAXON_KEY",
                "values": ["7952072", "2436693"],
            },
        ],
    }


@responses.activate
def test_submit_download_keeps_email_and_password_out_of_sanitized_result(
    client,
) -> None:
    responses.post(
        "https://api.gbif.org/v1/occurrence/download/request",
        body="0000000-260730120000000",
        status=201,
    )
    taxa = [TaxonMatch("Lepus europaeus", 7952072, "Lepus europaeus", 99)]

    submitted = client.submit_download(taxa)

    assert submitted["download_key"] == "0000000-260730120000000"
    serialized = json.dumps(submitted)
    assert "test-password" not in serialized
    assert "@" not in serialized
    request_payload = json.loads(responses.calls[0].request.body)
    assert request_payload["notificationAddresses"] == ["researcher@example.org"]
    assert request_payload["creator"] == "test-user"


@responses.activate
def test_successful_status_requires_download_link_doi_and_records(client) -> None:
    responses.get(
        "https://api.gbif.org/v1/occurrence/download/0000000-260730120000000",
        json={
            "key": "0000000-260730120000000",
            "status": "SUCCEEDED",
            "downloadLink": "https://api.gbif.org/file.zip",
            "doi": "10.15468/dl.example",
            "totalRecords": 123,
        },
    )

    status = client.download_status("0000000-260730120000000")

    assert status["doi"] == "10.15468/dl.example"
    assert status["total_records"] == 123


@responses.activate
def test_registry_metadata_uses_dataset_and_organization_keys(client) -> None:
    responses.get(
        "https://api.gbif.org/v1/dataset/dataset-key",
        json={"key": "dataset-key", "title": "Dataset title"},
    )
    responses.get(
        "https://api.gbif.org/v1/organization/org-key",
        json={"key": "org-key", "title": "Publisher title"},
    )

    assert client.dataset_metadata("dataset-key")["title"] == "Dataset title"
    assert client.organization_metadata("org-key")["title"] == "Publisher title"


@responses.activate
def test_download_archive_is_hashed_and_verified_as_zip(
    client,
    tmp_path: Path,
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("occurrence.txt", "gbifID\tdatasetKey\n1\tdataset-key\n")
    responses.get(
        "https://api.gbif.org/file.zip",
        body=archive_bytes.getvalue(),
        status=200,
    )

    result = client.retrieve_archive(
        {
            "download_key": "0000000-260730120000000",
            "download_url": "https://api.gbif.org/file.zip",
            "doi": "10.15468/dl.example",
            "total_records": 1,
        },
        tmp_path / "download.zip",
    )

    assert result["archive_sha256"]
    assert result["archive_files"] == ["occurrence.txt"]
    assert result["retrieved_at"].endswith("+00:00")
    assert (tmp_path / "download.zip").is_file()
