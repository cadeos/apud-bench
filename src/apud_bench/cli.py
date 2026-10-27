"""Command line entry point: inspect the subset registry and ground-truth files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from apud_bench import formats, registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="apud-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    list_cmd = commands.add_parser("list", help="list subsets")
    list_cmd.add_argument("--tier", choices=registry.TIERS)
    list_cmd.add_argument("--stratum", choices=registry.STRATA)
    commands.add_parser("check", help="validate the registry against the licence rules")
    commands.add_parser("stats", help="summarise the registry")
    inspect_cmd = commands.add_parser("inspect", help="read a PAGE/ALTO file as canvas JSON")
    inspect_cmd.add_argument("path")
    inspect_cmd.add_argument("--image-size", metavar="WxH", help="real page image size")
    args = parser.parse_args(argv)

    if args.command == "inspect":
        size = tuple(int(n) for n in args.image_size.split("x")) if args.image_size else None
        print(json.dumps(formats.load(args.path, size).to_dict(), ensure_ascii=False, indent=2))
        return 0

    subsets = registry.load()
    if args.command == "check":
        errors = registry.validate(subsets)
        for error in errors:
            print(error, file=sys.stderr)
        print(f"{len(subsets)} subsets, {len(errors)} errors")
        return 1 if errors else 0
    if args.command == "stats":
        for key in ("tier", "stratum", "script", "license"):
            counts = Counter(getattr(s, key) for s in subsets)
            print(f"{key}: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
        return 0
    for s in subsets:
        if (args.tier and s.tier != args.tier) or (args.stratum and s.stratum != args.stratum):
            continue
        print(f"{s.id:26} {s.tier:13} {s.license:18} {s.script:5} {s.centuries:8} {s.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
