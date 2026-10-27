"""Command line entry point: registry, ground-truth files, and the lock/fetch/build pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apud_bench import formats, iiif, registry, sources
from apud_bench.build import build
from apud_bench.fetch import data_dir, fetch

DEFAULT_BASE_URL = "http://localhost:8000"


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
    for name, help_text in (
        ("lock", "pin subset sources (GitHub commit, Zenodo version) in the lock file"),
        ("fetch", "download and extract pinned sources"),
        ("build", "fetch, then normalise pages into pages.jsonl and report.json"),
        ("iiif", "publish built pages as static IIIF (Level 0 tiles, manifest, ground truth)"),
    ):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("subsets", nargs="+", metavar="SUBSET")
        cmd.add_argument(
            "--data-dir",
            type=Path,
            help="cache directory (default: $APUD_BENCH_DATA or ~/.cache/apud-bench)",
        )
        cmd.set_defaults(data_dir=None)
        if name == "iiif":
            cmd.add_argument(
                "--base-url",
                default=DEFAULT_BASE_URL,
                help="public URL of <data-dir>/iiif (default: %(default)s, see `serve`)",
            )
    serve_cmd = commands.add_parser("serve", help="serve <data-dir>/iiif over HTTP with CORS")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "serve":
        _serve((args.data_dir or data_dir()) / "iiif", args.port)
        return 0

    if args.command == "inspect":
        size = tuple(int(n) for n in args.image_size.split("x")) if args.image_size else None
        print(json.dumps(formats.load(args.path, size).to_dict(), ensure_ascii=False, indent=2))
        return 0

    subsets = registry.load()
    if args.command in ("lock", "fetch", "build", "iiif"):
        return _pipeline(args, {s.id: s for s in subsets})
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


def _pipeline(args: argparse.Namespace, by_id: dict[str, registry.Subset]) -> int:
    unknown = [s for s in args.subsets if s not in by_id]
    if unknown:
        print(f"unknown subsets: {', '.join(unknown)}", file=sys.stderr)
        return 2
    held = [s for s in args.subsets if by_id[s].tier == "hold"]
    if held:
        print(f"subsets on hold cannot be used: {', '.join(held)}", file=sys.stderr)
        return 2
    root = args.data_dir or data_dir()
    lock = sources.read_lock()

    with sources.client() as http:
        if args.command == "lock":
            for subset_id in args.subsets:
                lock[subset_id] = sources.resolve(by_id[subset_id], http)
                print(f"{subset_id}: pinned {_describe(lock[subset_id])}")
            sources.write_lock(lock)
            return 0

        missing = [s for s in args.subsets if s not in lock]
        if missing:
            print(f"not in lock file, run `apud-bench lock` first: {', '.join(missing)}")
            return 2
        for subset_id in args.subsets:
            source = fetch(subset_id, lock[subset_id], http, root)
            print(f"{subset_id}: source ready at {source}")
            if args.command == "iiif":
                pages_file = root / "build" / subset_id / "pages.jsonl"
                if not pages_file.exists():
                    print(f"{subset_id}: not built yet, run `apud-bench build` first")
                    return 2
                subset = by_id[subset_id]
                count = iiif.publish(
                    subset_id,
                    subset.name,
                    subset.license,
                    pages_file,
                    source,
                    root / "iiif" / subset_id,
                    args.base_url,
                )
                print(f"{subset_id}: {count} canvases at {args.base_url}/{subset_id}/manifest.json")
            if args.command == "build":
                report = build(
                    by_id[subset_id], lock[subset_id], source, root / "build" / subset_id
                )
                skipped = ", ".join(f"{k} {v}" for k, v in report.skipped.items()) or "none"
                print(
                    f"{subset_id}: {report.pages} pages, {report.lines} lines "
                    f"({report.empty_lines} empty, {report.lines_out_of_bounds} out of bounds); "
                    f"skipped: {skipped}"
                )
    return 0


def _serve(directory: Path, port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    print(f"serving {directory} at http://localhost:{port}/")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _describe(pin: dict) -> str:
    if pin["kind"] == "github":
        return f"github {pin['repo']}@{pin['commit'][:10]}"
    return f"zenodo record {pin['record']} ({len(pin['files'])} files)"


if __name__ == "__main__":
    raise SystemExit(main())
