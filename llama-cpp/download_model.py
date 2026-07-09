#!/usr/bin/env python3
"""Download a Ximilar GGUF model artifact and print its local paths.

Used by the shell scripts in models/ before they invoke llama-cli:

    eval "$(uv run download_model.py MODEL_URL_GGUF_LFM2_VL_450M LiquidAI/LFM2-VL-450M)"
    # sets MODEL_GGUF and MMPROJ_GGUF

Two environment variables drive the download:
- MODEL_DOWNLOAD_DIRECTORY (required): base directory where models are
  downloaded and extracted; each model gets its own subfolder.
- The env var named by the first argument (required on first download):
  the model archive URL — copy it from your model's details page at
  https://app.ximilar.com/platform/vlm/tasks/ (the link is valid for 24
  hours). The archive contains the fine-tuned text GGUF and the mmproj
  (vision projector) GGUF.

Later runs find the extracted subfolder and skip the download entirely.
Progress goes to stderr; stdout carries only the eval-able variable lines.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

MODEL_DOWNLOAD_DIRECTORY_ENV = "MODEL_DOWNLOAD_DIRECTORY"
MODEL_DETAILS_PAGE = "https://app.ximilar.com/platform/vlm/tasks/"


def log(message: str) -> None:
    print(message, file=sys.stderr)


def discover_gguf_files(path: Path) -> tuple[Path, Path]:
    """Find (model_gguf, mmproj_gguf) under an extracted archive directory.

    The mmproj (vision projector) is the .gguf whose filename contains
    "mmproj"; the model is the single remaining .gguf. Searches the
    directory and one level of subdirectories.
    """
    ggufs = sorted(path.glob("*.gguf")) + sorted(path.glob("*/*.gguf"))
    mmprojs = [g for g in ggufs if "mmproj" in g.name.lower()]
    models = [g for g in ggufs if "mmproj" not in g.name.lower()]

    if len(models) != 1 or len(mmprojs) != 1:
        found = ", ".join(str(g) for g in ggufs) or "none"
        raise RuntimeError(
            f"Expected exactly one model .gguf and one mmproj .gguf under {path}.\n"
            f"Found: {found}"
        )
    return models[0], mmprojs[0]


def download_and_extract(download_url: str, output_path: Path) -> None:
    """Download the archive and extract it into output_path."""
    with tempfile.TemporaryDirectory(prefix="ximilar-vlm-download-") as staging_dir:
        filename = Path(urllib.parse.urlparse(download_url).path).name or "artifact.bin"
        archive_path = Path(staging_dir) / filename
        with urllib.request.urlopen(download_url) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(output_path)
        elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix in {".tgz", ".tar"}:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(output_path, filter="data")
        else:
            raise RuntimeError(f"Unsupported artifact type: {archive_path.name}")


def ensure_gguf_model(model_url_env: str, model_id: str) -> tuple[Path, Path]:
    """Return (model_gguf, mmproj_gguf) local paths, downloading on first use."""
    base_dir = os.environ.get(MODEL_DOWNLOAD_DIRECTORY_ENV)
    if not base_dir:
        raise RuntimeError(
            f"Environment variable {MODEL_DOWNLOAD_DIRECTORY_ENV} is not set.\n"
            f"Set it to the base directory where models should be downloaded and extracted\n"
            f"(each model gets its own subfolder), e.g.:\n"
            f"    export {MODEL_DOWNLOAD_DIRECTORY_ENV}=./stored"
        )

    cache_dir = Path(base_dir).expanduser().resolve() / f"{model_id.split('/')[-1]}-GGUF"
    if cache_dir.is_dir():
        try:
            model_gguf, mmproj = discover_gguf_files(cache_dir)
            log(f"Using cached model: {model_gguf}")
            return model_gguf, mmproj
        except RuntimeError:
            log(f"Cached model at {cache_dir} is incomplete — re-downloading")

    download_url = os.environ.get(model_url_env)
    if not download_url:
        raise RuntimeError(
            f"Environment variable {model_url_env} is not set.\n"
            f"Set it to your model's download URL, e.g.:\n"
            f"    export {model_url_env}='https://...'\n"
            f"You can copy the URL from the model details page at {MODEL_DETAILS_PAGE}\n"
            f"(open your task, click the download action on the trained model, then Copy link)."
        )

    log(f"Downloading model to {cache_dir} ...")
    download_and_extract(download_url, cache_dir)
    model_gguf, mmproj = discover_gguf_files(cache_dir)
    log(f"Model downloaded: {model_gguf}")
    return model_gguf, mmproj


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <MODEL_URL_ENV_VAR> <MODEL_ID>")

    try:
        model_gguf, mmproj = ensure_gguf_model(sys.argv[1], sys.argv[2])
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}")

    print(f"MODEL_GGUF={shlex.quote(str(model_gguf))}")
    print(f"MMPROJ_GGUF={shlex.quote(str(mmproj))}")


if __name__ == "__main__":
    main()
