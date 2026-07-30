"""Citable GBIF taxon resolution, download, and registry operations."""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_ROOT = "https://api.gbif.org"
TIMEOUT = (10, 60)
EXCLUDED_TOKENS = ("bat", "pipistrell")


@dataclass(frozen=True)
class TaxonMatch:
    requested_name: str
    taxon_key: int
    accepted_name: str
    confidence: int


def build_download_predicate(taxa: list[TaxonMatch]) -> dict[str, object]:
    """Build the frozen Great Britain 2022–2025 occurrence predicate."""

    if not taxa:
        raise ValueError("at least one accepted taxon is required")
    names = " ".join(
        f"{taxon.requested_name} {taxon.accepted_name}" for taxon in taxa
    ).casefold()
    if any(token in names for token in EXCLUDED_TOKENS):
        raise ValueError("download taxa contain an excluded bat taxon")
    keys = list(dict.fromkeys(str(taxon.taxon_key) for taxon in taxa))
    return {
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
            {"type": "in", "key": "TAXON_KEY", "values": keys},
        ],
    }


class GBIFClient:
    """Small, explicit boundary around public GBIF APIs."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        notification_email: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.username = username
        self.password = password
        self.notification_email = notification_email
        self.session = session or requests.Session()
        retries = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET", "HEAD")),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update(
            {"User-Agent": "provenance-sdm/0.1 (Ecological Modelling study)"}
        )

    @classmethod
    def from_environment(cls) -> "GBIFClient":
        return cls(
            username=os.environ.get("GBIF_USERNAME"),
            password=os.environ.get("GBIF_PASSWORD"),
            notification_email=os.environ.get("GBIF_NOTIFICATION_EMAIL"),
        )

    def _get_json(self, path: str, **params: object) -> dict[str, object]:
        response = self.session.get(
            f"{API_ROOT}{path}",
            params=params or None,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"GBIF returned a non-object response for {path}")
        return payload

    def resolve_taxon(self, scientific_name: str) -> TaxonMatch:
        """Resolve one name using GBIF's current v2 species-match service."""

        if not scientific_name or any(
            token in scientific_name.casefold() for token in EXCLUDED_TOKENS
        ):
            raise ValueError("scientific name is empty or excluded")
        payload = self._get_json(
            "/v2/species/match",
            scientificName=scientific_name,
        )
        usage = payload.get("usage")
        diagnostics = payload.get("diagnostics", {})
        if not isinstance(usage, dict) or not isinstance(diagnostics, dict):
            raise ValueError(f"GBIF did not match {scientific_name!r}")
        if usage.get("status") != "ACCEPTED" or usage.get("rank") != "SPECIES":
            raise ValueError(
                f"GBIF match for {scientific_name!r} is not an accepted species"
            )
        try:
            key = int(usage["key"])
            accepted_name = str(usage["name"])
            confidence = int(diagnostics["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"GBIF match for {scientific_name!r} is incomplete"
            ) from exc
        return TaxonMatch(scientific_name, key, accepted_name, confidence)

    def submit_download(self, taxa: list[TaxonMatch]) -> dict[str, object]:
        """Submit an authenticated asynchronous occurrence download."""

        if not self.username or not self.password or not self.notification_email:
            raise ValueError(
                "GBIF_USERNAME, GBIF_PASSWORD, and GBIF_NOTIFICATION_EMAIL "
                "are required to submit a download"
            )
        predicate = build_download_predicate(taxa)
        payload = {
            "creator": self.username,
            "notificationAddresses": [self.notification_email],
            "sendNotification": True,
            "format": "SIMPLE_CSV",
            "predicate": predicate,
        }
        response = self.session.post(
            f"{API_ROOT}/v1/occurrence/download/request",
            json=payload,
            auth=(self.username, self.password),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        download_key = response.text.strip().strip('"')
        if not download_key:
            raise ValueError("GBIF download request returned an empty key")
        return {
            "download_key": download_key,
            "format": "SIMPLE_CSV",
            "predicate": predicate,
            "taxa": [asdict(taxon) for taxon in taxa],
        }

    def download_status(self, download_key: str) -> dict[str, object]:
        """Return sanitized status and validate completed download metadata."""

        payload = self._get_json(f"/v1/occurrence/download/{download_key}")
        status = str(payload.get("status", ""))
        result: dict[str, object] = {
            "download_key": str(payload.get("key", download_key)),
            "status": status,
        }
        if status != "SUCCEEDED":
            return result
        download_url = payload.get("downloadLink")
        doi = payload.get("doi")
        total_records = payload.get("totalRecords")
        if (
            not isinstance(download_url, str)
            or not download_url
            or not isinstance(doi, str)
            or not doi
            or not isinstance(total_records, int)
            or total_records <= 0
        ):
            raise ValueError("successful GBIF download lacks URL, DOI, or records")
        result.update(
            {
                "download_url": download_url,
                "doi": doi,
                "total_records": total_records,
            }
        )
        return result

    def dataset_metadata(self, dataset_key: str) -> dict[str, object]:
        return self._get_json(f"/v1/dataset/{dataset_key}")

    def organization_metadata(self, organization_key: str) -> dict[str, object]:
        return self._get_json(f"/v1/organization/{organization_key}")

    def retrieve_archive(
        self,
        completed_status: dict[str, object],
        path: Path,
    ) -> dict[str, object]:
        """Stream, hash, and ZIP-verify a completed GBIF archive."""

        download_url = completed_status.get("download_url")
        if completed_status.get("doi") is None or not isinstance(download_url, str):
            raise ValueError("archive retrieval requires completed DOI status")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        response = self.session.get(download_url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    digest.update(chunk)
                    stream.write(chunk)
        try:
            with zipfile.ZipFile(temporary) as archive:
                bad_member = archive.testzip()
                files = archive.namelist()
        except zipfile.BadZipFile as exc:
            temporary.unlink(missing_ok=True)
            raise ValueError("GBIF archive is not a valid ZIP file") from exc
        if bad_member is not None:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"GBIF archive member failed CRC: {bad_member}")
        os.replace(temporary, destination)
        return {
            **completed_status,
            "archive_sha256": digest.hexdigest(),
            "archive_files": files,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
