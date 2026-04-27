"""Helpers for downloading trained model artifacts from the Ximilar backend."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


VALID_MARKERS = ("adapter_config.json", "model.pt", "config.json")


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def request_model_download(model_uuid: str, api_url: str, api_token: str) -> dict:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/model/{model_uuid}/request-download",
        headers={"Authorization": f"Token {api_token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Backend request failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Backend request failed: {exc.reason}") from exc

    if "download_url" not in payload:
        raise RuntimeError("Backend response is missing download_url")
    return payload


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Artifact download failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Artifact download failed: {exc.reason}") from exc
    return destination


def extract_artifact(archive_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    suffixes = archive_path.suffixes
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(output_path)
    elif suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix in {".tgz", ".tar"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_path, filter="data")
    else:
        raise RuntimeError(f"Unsupported artifact type: {archive_path.name}")
    return output_path


def resolve_model_dir(path: Path) -> Path:
    if any((path / marker).exists() for marker in VALID_MARKERS):
        return path

    child_dirs = [child for child in path.iterdir() if child.is_dir()]
    if len(child_dirs) == 1 and any((child_dirs[0] / marker).exists() for marker in VALID_MARKERS):
        return child_dirs[0]

    return path


def validate_model_dir(path: Path) -> None:
    resolved = resolve_model_dir(path)
    if any((resolved / marker).exists() for marker in VALID_MARKERS):
        return
    markers = ", ".join(VALID_MARKERS)
    raise RuntimeError(f"Extracted directory does not look like a model directory. Expected one of: {markers}")


def download_and_extract_model(model_uuid: str, output_path: Path, api_url: str, api_token: str) -> Path:
    payload = request_model_download(model_uuid=model_uuid, api_url=api_url, api_token=api_token)
    download_url = payload["download_url"]

    staging_dir = Path(tempfile.gettempdir()) / "ximilar-vlm-downloads" / model_uuid
    staging_dir.mkdir(parents=True, exist_ok=True)

    parsed_url = urllib.parse.urlparse(download_url)
    filename = Path(parsed_url.path).name or "artifact.bin"
    archive_path = staging_dir / filename
    download_file(download_url, archive_path)

    extracted_root = extract_artifact(archive_path, output_path)
    model_dir = resolve_model_dir(extracted_root)
    validate_model_dir(model_dir)
    return model_dir
