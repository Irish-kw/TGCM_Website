"""Checksum-verified access to the compressed reproduction assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def artifact_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "tgcm_review").is_dir() and (candidate / "data").is_dir():
            return candidate
    package_root = Path(__file__).resolve().parents[1]
    if (package_root / "tgcm_review").is_dir() and (package_root / "data").is_dir():
        return package_root
    raise FileNotFoundError("Could not locate the TGCM reproduction-artifact root")


# Compatibility name used by the paper-numbered experiment entry points.
reviewer_root = artifact_root


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path | None = None) -> dict:
    base = root or artifact_root()
    path = base / "data" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _download_if_needed(part: dict, data_dir: Path) -> Path:
    destination = data_dir / part["path"]
    if destination.is_file() and _sha256(destination) == part["sha256"]:
        return destination
    url = part.get("url")
    if not url:
        if destination.exists():
            raise ValueError(f"Checksum mismatch for local asset part: {destination}")
        raise FileNotFoundError(
            f"Missing asset part {destination}. Its public download URL has not been added yet."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not token and part.get("api_url") and (gh := shutil.which("gh")):
            completed = subprocess.run(
                [gh, "auth", "token"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                token = completed.stdout.strip()
        if token and part.get("api_url"):
            request = urllib.request.Request(
                part["api_url"],
                headers={
                    "Accept": "application/octet-stream",
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        else:
            urllib.request.urlretrieve(url, temporary)
        if _sha256(temporary) != part["sha256"]:
            raise ValueError(f"Downloaded checksum mismatch: {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, mode="r:xz") as bundle:
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_resolved not in (member_path, *member_path.parents):
                raise ValueError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)


def prepare_asset(name: str, root: Path | None = None, force: bool = False) -> Path:
    """Verify, join, and unpack one named asset into ``data/extracted``."""

    base = root or artifact_root()
    data_dir = base / "data"
    manifest = load_manifest(base)
    try:
        spec = manifest["assets"][name]
    except KeyError as exc:
        raise KeyError(f"Unknown asset {name!r}; choices={sorted(manifest.get('assets', {}))}") from exc

    destination = data_dir / "extracted" / name
    ready = destination / ".ready.json"
    if ready.is_file() and not force:
        recorded = json.loads(ready.read_text(encoding="utf-8"))
        if recorded.get("archive_sha256") == spec["archive_sha256"]:
            return destination
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    parts = [_download_if_needed(part, data_dir) for part in spec["parts"]]
    cache_dir = data_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    joined = cache_dir / f"{name}.tar.xz"
    with joined.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if _sha256(joined) != spec["archive_sha256"]:
        joined.unlink(missing_ok=True)
        raise ValueError(f"Joined archive checksum mismatch for asset {name}")
    _safe_extract(joined, destination)
    ready.write_text(
        json.dumps({"asset": name, "archive_sha256": spec["archive_sha256"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
