"""Download pinned sources into a local cache and extract them safely.

Downloaded archives are untrusted input: extraction refuses paths that escape the target
directory, links, and archives that would expand beyond a size cap.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import httpx

from apud_bench.sources import SourceError

MAX_EXTRACTED_BYTES = 60 * 1024**3
_MARKER = ".apud-source.json"


def data_dir() -> Path:
    return Path(os.environ.get("APUD_BENCH_DATA", Path.home() / ".cache" / "apud-bench"))


def source_dir(subset_id: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / "sources" / subset_id


def fetch(
    subset_id: str, pin: dict[str, Any], http: httpx.Client, root: Path | None = None
) -> Path:
    """Make ``sources/<subset_id>`` match ``pin``; a no-op when it already does."""
    root = root or data_dir()
    target = source_dir(subset_id, root)
    marker = target / _MARKER
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) == pin:
        return target

    downloads = root / "downloads" / subset_id
    downloads.mkdir(parents=True, exist_ok=True)
    if pin["kind"] == "github":
        archives = [_download(pin["url"], downloads / f"{pin['commit']}.tar.gz", None, http)]
    elif pin["kind"] == "zenodo":
        archives = [
            _download(f["url"], downloads / f["key"], f["checksum"], http) for f in pin["files"]
        ]
    else:
        raise SourceError(f"{subset_id}: unknown source kind {pin['kind']!r}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for archive in archives:
        extract(archive, target)
    marker.write_text(json.dumps(pin, indent=2), encoding="utf-8")
    return target


def _download(url: str, dest: Path, checksum: str | None, http: httpx.Client) -> Path:
    if dest.exists() and (checksum is None or _verify(dest, checksum)):
        return dest
    partial = dest.with_name(dest.name + ".part")
    with http.stream("GET", url) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    if checksum and not _verify(partial, checksum):
        partial.unlink()
        raise SourceError(f"checksum mismatch for {url}")
    partial.replace(dest)
    return dest


def _verify(path: Path, checksum: str) -> bool:
    algorithm, _, expected = checksum.partition(":")
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def extract(archive: Path, target: Path) -> None:
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tar:
            total = sum(m.size for m in tar.getmembers() if m.isfile())
            _check_size(archive, total)
            tar.extractall(target, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _check_size(archive, sum(i.file_size for i in zf.infolist()))
            root = target.resolve()
            for info in zf.infolist():
                destination = (target / info.filename).resolve()
                if not destination.is_relative_to(root):
                    raise SourceError(f"{archive.name}: unsafe path {info.filename!r}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue  # symlink entry
                zf.extract(info, target)
    else:
        shutil.copy2(archive, target / archive.name)


def _check_size(archive: Path, total: int) -> None:
    if total > MAX_EXTRACTED_BYTES:
        raise SourceError(f"{archive.name} expands to {total} bytes, over the safety cap")
