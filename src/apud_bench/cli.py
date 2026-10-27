"""Command line entry point: registry, ground-truth files, and the lock/fetch/build pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apud_bench import evaluate, formats, iiif, registry, sources, splits, tasks
from apud_bench.build import build
from apud_bench.fetch import data_dir, fetch

DEFAULT_BASE_URL = "http://localhost:8000"
RUN_CONDITIONS = ("transcription:region", "transcription:page", "grounding:page")


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
    split_cmd = commands.add_parser("split", help="draw and freeze the evaluation split")
    split_cmd.add_argument("subsets", nargs="+", metavar="SUBSET")
    split_cmd.add_argument("--pages", type=int, default=20, help="pages per subset (%(default)s)")
    split_cmd.add_argument("--seed", type=int, default=0)
    split_cmd.add_argument("--data-dir", type=Path, default=None)
    tasks_cmd = commands.add_parser("tasks", help="write transcription and grounding items")
    tasks_cmd.add_argument("subsets", nargs="+", metavar="SUBSET")
    tasks_cmd.add_argument("--out", type=Path, required=True, help="items JSONL to write")
    tasks_cmd.add_argument("--base-url", default=DEFAULT_BASE_URL)
    tasks_cmd.add_argument("--data-dir", type=Path, default=None)
    tasks_cmd.add_argument(
        "--task", choices=("transcription", "grounding"), action="append", dest="task_names"
    )
    baseline_cmd = commands.add_parser("baseline", help="write oracle or trivial predictions")
    baseline_cmd.add_argument("--items", type=Path, required=True)
    baseline_cmd.add_argument("--kind", choices=("oracle", "trivial"), required=True)
    baseline_cmd.add_argument("--out", type=Path, required=True)
    eval_cmd = commands.add_parser("evaluate", help="score predictions against items")
    eval_cmd.add_argument("--items", type=Path, required=True)
    eval_cmd.add_argument("--predictions", type=Path, required=True)
    eval_cmd.add_argument("--task", choices=("transcription", "grounding"), help="score one task")
    run_cmd = commands.add_parser("run", help="query a model on items (needs the runners extra)")
    run_cmd.add_argument("--items", type=Path, required=True)
    run_cmd.add_argument("--condition", required=True, choices=RUN_CONDITIONS)
    run_cmd.add_argument("--out", type=Path, required=True, help="run directory (resumable)")
    run_cmd.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    run_cmd.add_argument("--model", help="model id (default: claude-opus-5-5 for anthropic)")
    run_cmd.add_argument("--effort", default="medium", help="anthropic effort (%(default)s)")
    run_cmd.add_argument("--base-url", help="openai-compatible endpoint URL")
    run_cmd.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var with the key")
    run_cmd.add_argument("--max-side", type=int, help="override the image long-edge limit")
    run_cmd.add_argument("--group-size", type=int, default=10, help="lines per page call")
    run_cmd.add_argument("--pages-per-subset", type=int, help="only the first N pages")
    run_cmd.add_argument("--workers", type=int, default=4)
    run_cmd.add_argument("--dry-run", action="store_true", help="fetch images, estimate cost")
    run_cmd.add_argument("--price-in", type=float, default=4.0, help="$ per 1M input tokens")
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

    if args.command == "run":
        return _run(args)

    if args.command in ("baseline", "evaluate"):
        items = _read_jsonl(args.items)
        if getattr(args, "task", None):
            items = [i for i in items if i["task"] == args.task]
        if args.command == "baseline":
            _write_jsonl(args.out, evaluate.baseline(items, args.kind))
            return 0
        predictions = {p["id"]: p for p in _read_jsonl(args.predictions)}
        print(json.dumps(evaluate.score(items, predictions), indent=2))
        return 0

    subsets = registry.load()
    if args.command in ("split", "tasks"):
        return _evaluation_data(args, {s.id: s for s in subsets})
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


def _evaluation_data(args: argparse.Namespace, by_id: dict[str, registry.Subset]) -> int:
    unknown = [s for s in args.subsets if s not in by_id]
    if unknown:
        print(f"unknown subsets: {', '.join(unknown)}", file=sys.stderr)
        return 2
    root = args.data_dir or data_dir()
    lock = sources.read_lock()
    all_items: list[dict] = []
    for subset_id in args.subsets:
        pages_file = root / "build" / subset_id / "pages.jsonl"
        if subset_id not in lock or not pages_file.exists():
            print(f"{subset_id}: not built yet, run `apud-bench build` first", file=sys.stderr)
            return 2
        if args.command == "split":
            built = [json.loads(line)["xml_path"] for line in pages_file.open(encoding="utf-8")]
            chosen = splits.draw(built, args.pages, args.seed)
            path = splits.write(subset_id, lock[subset_id], chosen, args.seed)
            documents = len({splits.document_of(p) for p in chosen})
            print(f"{subset_id}: {len(chosen)} pages from {documents} documents -> {path}")
            continue
        chosen = splits.read(subset_id, lock[subset_id])
        pages = tasks.select_pages(pages_file, chosen)
        names = tuple(args.task_names or ("transcription", "grounding"))
        subset_items = tasks.items(subset_id, pages, args.base_url, names)
        counts = Counter(i["task"] for i in subset_items)
        print(f"{subset_id}: " + ", ".join(f"{n} {counts[n]}" for n in names))
        all_items.extend(subset_items)
    if args.command == "tasks":
        _write_jsonl(args.out, all_items)
        print(f"{len(all_items)} items -> {args.out}")
    return 0


def _run(args: argparse.Namespace) -> int:
    # Imported lazily: model SDKs and apud are optional (`pip install apud-bench[runners]`).
    from apud_bench.runners import run as runner

    if args.provider == "anthropic":
        from apud_bench.runners.claude import DEFAULT_MODEL, ClaudeProvider

        provider = ClaudeProvider(args.model or DEFAULT_MODEL, args.effort)
    else:
        from apud_bench.runners.openai_compatible import OpenAICompatibleProvider

        if not args.model:
            print("--model is required for the openai provider", file=sys.stderr)
            return 2
        provider = OpenAICompatibleProvider(args.model, args.base_url, args.api_key_env)
    if args.max_side:
        provider.max_side = args.max_side
    items = _read_jsonl(args.items)
    languages = {s.id: s.langs for s in registry.load()}
    jobs = runner.plan(items, args.condition, args.group_size, args.pages_per_subset)
    summary = runner.run(
        provider, jobs, runner.ImageSource(), args.out, languages, args.workers, args.dry_run
    )
    if args.dry_run:
        tokens = summary["estimated_image_tokens"] + summary["estimated_prompt_tokens"]
        summary["estimated_input_cost_usd"] = round(tokens / 1e6 * args.price_in, 2)
        summary["note"] = "estimate covers input only; output and thinking tokens are extra"
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] else 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


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
