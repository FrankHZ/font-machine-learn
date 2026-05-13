from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.review import ReviewBaseline, export_review_report


def baseline_spec(value: str) -> ReviewBaseline:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must use NAME=PATH")
    name, path = value.split("=", 1)
    return ReviewBaseline(name=name, metadata_json=Path(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build side-by-side baseline review artifacts.")
    parser.add_argument("--source-metadata", type=Path, default=Path("data/processed/glyphs/source_metadata.json"))
    parser.add_argument("--baseline", action="append", type=baseline_spec, default=None)
    parser.add_argument("--review-json", type=Path, default=Path("data/processed/glyphs/review_report.json"))
    parser.add_argument("--worst-cases-json", type=Path, default=Path("data/processed/glyphs/review_worst_cases.json"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("data/processed/glyphs/review_contact.png"))
    parser.add_argument("--worst-count", type=int, default=160)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_review_report(
        source_metadata=args.source_metadata,
        baselines=args.baseline,
        review_json=args.review_json,
        worst_cases_json=args.worst_cases_json,
        contact_sheet=args.contact_sheet,
        worst_count=args.worst_count,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
