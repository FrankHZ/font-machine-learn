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

from font_machine_learn.paths import (
    SOURCE_METADATA,
    WQY_ALIGNMENT_CONTACT,
    WQY_ALIGNMENT_DIR,
    WQY_ALIGNMENT_METADATA,
    WQY_ALIGNMENT_SEARCH,
)
from font_machine_learn.wqy_alignment import export_wqy_alignment_diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search WQY source alignment/weight against target mask choices.")
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=WQY_ALIGNMENT_DIR)
    parser.add_argument("--metadata", type=Path, default=WQY_ALIGNMENT_METADATA)
    parser.add_argument("--search-json", type=Path, default=WQY_ALIGNMENT_SEARCH)
    parser.add_argument("--contact-sheet", type=Path, default=WQY_ALIGNMENT_CONTACT)
    parser.add_argument("--search-limit", type=int, default=0)
    parser.add_argument("--contact-count", type=int, default=96)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_wqy_alignment_diagnostic(
        source_metadata=args.source_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        search_json=args.search_json,
        contact_sheet=args.contact_sheet,
        search_limit=None if args.search_limit <= 0 else args.search_limit,
        contact_count=args.contact_count,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
