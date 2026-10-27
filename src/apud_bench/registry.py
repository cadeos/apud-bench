"""Registry of benchmark subsets and the licence rules that decide their tier.

Data is never redistributed: each subset is fetched from its original source, and everything the
benchmark derives from it (normalised geometry, splits, manifests) carries that subset's licence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Tier = Literal["core", "nc-extension", "hold"]
Task = Literal["T", "G"]  # T = transcription, G = grounding

TIERS: tuple[str, ...] = ("core", "nc-extension", "hold")
STRATA = ("ancient", "medieval", "early-modern", "late", "mixed")
CORE_LICENCES = frozenset(
    {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "Apache-2.0", "MIT", "Etalab-2.0"}
)


@dataclass(frozen=True)
class Subset:
    id: str
    name: str
    url: str
    license: str
    tier: Tier
    stratum: str
    script: str
    langs: list[str]
    centuries: str
    tasks: list[Task]
    flags: list[str] = field(default_factory=list)
    licence_note: str | None = None
    fetch: dict[str, Any] | None = None
    build: dict[str, Any] | None = None

    @property
    def share_alike(self) -> bool:
        return "-SA-" in self.license


def default_path() -> Path:
    packaged = Path(__file__).with_name("subsets.yaml")
    if packaged.exists():
        return packaged
    return Path(__file__).parents[2] / "subsets" / "subsets.yaml"


def load(path: Path | None = None) -> list[Subset]:
    raw = yaml.safe_load((path or default_path()).read_text(encoding="utf-8"))
    return [Subset(**entry) for entry in raw["subsets"]]


def validate(subsets: list[Subset]) -> list[str]:
    """Return every rule violation; an empty list means the registry is consistent."""
    errors: list[str] = []
    seen: set[str] = set()
    for s in subsets:
        if s.id in seen:
            errors.append(f"{s.id}: duplicate id")
        seen.add(s.id)
        if s.tier not in TIERS:
            errors.append(f"{s.id}: unknown tier {s.tier!r}")
        if s.stratum not in STRATA:
            errors.append(f"{s.id}: unknown stratum {s.stratum!r}")
        if not s.tasks or not set(s.tasks) <= {"T", "G"}:
            errors.append(f"{s.id}: tasks must be a non-empty subset of T, G")
        if s.tier == "core" and s.license not in CORE_LICENCES:
            errors.append(f"{s.id}: licence {s.license} is not allowed in core")
        if s.tier == "nc-extension" and "-NC-" not in s.license:
            errors.append(f"{s.id}: nc-extension subset without a NonCommercial licence")
        if s.tier != "nc-extension" and "-NC-" in s.license:
            errors.append(f"{s.id}: NonCommercial licence outside nc-extension")
    return errors
