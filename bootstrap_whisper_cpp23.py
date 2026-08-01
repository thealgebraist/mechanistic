#!/usr/bin/env python3
"""Reconstruct and verify the pinned local assets for the Whisper C++23 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "whisper_cpp23_assets.json"
REQUIREMENTS_PATH = ROOT / "requirements-whisper.lock"
REPORT_PATH = ROOT / "work/whisper_cpp23_bootstrap_report.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def asset_url(asset: dict) -> str:
    prefix = "datasets/" if asset["repository_type"] == "dataset" else ""
    repository = urllib.parse.quote(asset["repository"], safe="/")
    revision = urllib.parse.quote(asset["revision"], safe="")
    filename = urllib.parse.quote(asset["filename"], safe="/")
    return f"https://huggingface.co/{prefix}{repository}/resolve/{revision}/{filename}?download=true"


def copy_or_download(asset: dict, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        if asset["repository_type"] == "repository_copy":
            source = ROOT / asset["filename"]
            if not source.is_file():
                raise RuntimeError(f"repository fixture missing: {asset['filename']}")
            shutil.copyfile(source, temporary)
            origin = asset["filename"]
        else:
            origin = asset_url(asset)
            request = urllib.request.Request(
                origin, headers={"User-Agent": "mechanistic-whisper-bootstrap/1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, \
                    temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = digest(temporary)
        if actual != asset["sha256"]:
            raise RuntimeError(
                f"download hash mismatch for {asset['destination']}: "
                f"expected {asset['sha256']}, got {actual}"
            )
        os.replace(temporary, destination)
        return origin
    finally:
        temporary.unlink(missing_ok=True)


def install_environment(environment: Path) -> dict:
    python = environment / "bin/python"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--requirement", str(REQUIREMENTS_PATH)],
        cwd=ROOT,
        check=True,
    )
    probe = (
        "import json,torch,transformers,huggingface_hub,numpy,safetensors,pyarrow;"
        "print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,"
        "'huggingface_hub':huggingface_hub.__version__,'numpy':numpy.__version__,"
        "'safetensors':safetensors.__version__,'pyarrow':pyarrow.__version__}))"
    )
    return json.loads(subprocess.check_output([str(python), "-c", probe], text=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight", action="store_true", help="verify only; never download or install"
    )
    parser.add_argument(
        "--repair", action="store_true", help="replace present files with wrong hashes"
    )
    parser.add_argument(
        "--install-python", action="store_true", help="create work/venv and install pinned packages"
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = []
    failed = False
    for asset in manifest["assets"]:
        destination = ROOT / asset["destination"]
        status = "MISSING"
        actual = None
        origin = None
        if destination.is_file():
            actual = digest(destination)
            status = "VERIFIED" if actual == asset["sha256"] else "HASH_MISMATCH"
        if status == "MISSING" and not arguments.preflight:
            origin = copy_or_download(asset, destination)
            actual = digest(destination)
            status = "DOWNLOADED"
        elif status == "HASH_MISMATCH" and arguments.repair and not arguments.preflight:
            origin = copy_or_download(asset, destination)
            actual = digest(destination)
            status = "REPAIRED"
        if actual != asset["sha256"]:
            failed = True
        row = {
            "destination": asset["destination"],
            "status": status,
            "expected_sha256": asset["sha256"],
            "actual_sha256": actual,
        }
        if origin:
            row["origin"] = origin
        rows.append(row)
        if not arguments.quiet:
            print(f"{status:13} {asset['destination']}")
    versions = None
    if arguments.install_python:
        if arguments.preflight:
            parser.error("--preflight and --install-python are mutually exclusive")
        versions = install_environment(ROOT / "work/venv")
    report = {
        "certificate": "WHISPER_CPP23_BOOTSTRAP_PREFLIGHT_1",
        "asset_manifest_schema": manifest["schema"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "asset_count": len(rows),
        "all_assets_verified": not failed,
        "downloaded_or_repaired": sum(
            row["status"] in {"DOWNLOADED", "REPAIRED"} for row in rows
        ),
        "python_environment_versions": versions,
        "assets": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        print("WHISPER_CPP23_BOOTSTRAP_FAILED", file=sys.stderr)
        return 1
    print(
        "WHISPER_CPP23_BOOTSTRAP_OK "
        f"assets={len(rows)} downloaded_or_repaired={report['downloaded_or_repaired']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
