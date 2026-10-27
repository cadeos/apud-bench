"""Resolve subset URLs to pinned, downloadable sources and keep them in a lock file.

A pin is what makes a build reproducible: a GitHub commit SHA, or a Zenodo record version
with per-file checksums. ``apud-bench lock`` writes pins to ``subsets/sources.lock.json``;
``fetch`` and ``build`` only ever use what the lock says.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from apud_bench import __version__
from apud_bench.registry import Subset

USER_AGENT = f"apud-bench/{__version__} (+https://github.com/cadeos/apud-bench)"
METADATA_SUFFIXES = (".cff", ".yml", ".yaml", ".md")

_GITHUB = re.compile(r"github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/|$)")
_ZENODO = re.compile(r"zenodo\.org/records?/(\d+)|10\.5281/zenodo\.(\d+)")


class SourceError(RuntimeError):
    pass


def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=httpx.Timeout(60.0)
    )


def get(http: httpx.Client, url: str, retries: int = 5) -> httpx.Response:
    """GET with exponential backoff on rate limiting and server errors (Zenodo returns 429)."""
    for attempt in range(retries):
        response = http.get(url)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
    response.raise_for_status()
    return response


def resolve(subset: Subset, http: httpx.Client) -> dict[str, Any]:
    if match := _GITHUB.search(subset.url):
        return _resolve_github(f"{match.group(1)}/{match.group(2)}", http)
    if match := _ZENODO.search(subset.url):
        return _resolve_zenodo(match.group(1) or match.group(2), subset, http)
    raise SourceError(f"{subset.id}: unsupported source {subset.url}")


def _resolve_github(repo: str, http: httpx.Client) -> dict[str, Any]:
    info = get(http, f"https://api.github.com/repos/{repo}").json()
    branch = info["default_branch"]
    commit = get(http, f"https://api.github.com/repos/{repo}/commits/{branch}").json()["sha"]
    return {
        "kind": "github",
        "repo": repo,
        "commit": commit,
        "url": f"https://codeload.github.com/{repo}/tar.gz/{commit}",
    }


def _resolve_zenodo(record_id: str, subset: Subset, http: httpx.Client) -> dict[str, Any]:
    # A concept DOI resolves to the latest version; the lock pins that exact version.
    record = get(http, f"https://zenodo.org/api/records/{record_id}").json()
    files = {f["key"]: f for f in record["files"]}
    wanted = (subset.fetch or {}).get("files")
    if wanted:
        missing = sorted(set(wanted) - files.keys())
        if missing:
            raise SourceError(f"{subset.id}: files {missing} not in Zenodo record {record['id']}")
        keys = list(wanted)
    else:
        keys = [k for k in files if not k.lower().endswith(METADATA_SUFFIXES)]
    return {
        "kind": "zenodo",
        "record": record["id"],
        "files": [
            {
                "key": k,
                "size": files[k]["size"],
                "checksum": files[k]["checksum"],
                "url": files[k]["links"]["self"],
            }
            for k in keys
        ],
    }


def lock_path() -> Path:
    return Path(__file__).parents[2] / "subsets" / "sources.lock.json"


def read_lock(path: Path | None = None) -> dict[str, Any]:
    path = path or lock_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_lock(lock: dict[str, Any], path: Path | None = None) -> None:
    path = path or lock_path()
    text = json.dumps(dict(sorted(lock.items())), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
