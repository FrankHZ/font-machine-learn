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

from font_machine_learn.boundary_rules import export_boundary_rules
from font_machine_learn.paths import (
    BOUNDARY_RULE_CONTACT,
    BOUNDARY_RULE_DIR,
    BOUNDARY_RULE_METADATA,
    BOUNDARY_RULE_SEARCH,
    BOUNDARY_RULE_WORST_CONTACT,
    TARGET_METADATA,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search explainable ge2 boundary rules for target level 2 vs 3.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=BOUNDARY_RULE_DIR)
    parser.add_argument("--metadata", type=Path, default=BOUNDARY_RULE_METADATA)
    parser.add_argument("--search-json", type=Path, default=BOUNDARY_RULE_SEARCH)
    parser.add_argument("--contact-sheet", type=Path, default=BOUNDARY_RULE_CONTACT)
    parser.add_argument("--worst-contact-sheet", type=Path, default=BOUNDARY_RULE_WORST_CONTACT)
    parser.add_argument("--search-limit", type=int, default=0)
    parser.add_argument("--worst-count", type=int, default=160)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_boundary_rules(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        search_json=args.search_json,
        contact_sheet=args.contact_sheet,
        worst_contact_sheet=args.worst_contact_sheet,
        search_limit=None if args.search_limit <= 0 else args.search_limit,
        worst_count=args.worst_count,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
