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

from font_machine_learn.weight_search import export_weight_search
from font_machine_learn.paths import (
    BASELINE_WEIGHTED_SHADOW_CONTACT,
    BASELINE_WEIGHTED_SHADOW_DIR,
    SOURCE_METADATA,
    SOURCE_WEIGHTED_DIR,
    WEIGHT_SEARCH_JSON,
    WEIGHT_SEARCH_METADATA,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search source glyph weight/offset before shadowing.")
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--source-out-dir", type=Path, default=SOURCE_WEIGHTED_DIR)
    parser.add_argument("--shadow-out-dir", type=Path, default=BASELINE_WEIGHTED_SHADOW_DIR)
    parser.add_argument("--metadata", type=Path, default=WEIGHT_SEARCH_METADATA)
    parser.add_argument("--search-json", type=Path, default=WEIGHT_SEARCH_JSON)
    parser.add_argument("--contact-sheet", type=Path, default=BASELINE_WEIGHTED_SHADOW_CONTACT)
    parser.add_argument("--search-limit", type=int, default=512)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_weight_search(
        source_metadata=args.source_metadata,
        source_out_dir=args.source_out_dir,
        shadow_out_dir=args.shadow_out_dir,
        metadata_json=args.metadata,
        search_json=args.search_json,
        contact_sheet=args.contact_sheet,
        search_limit=None if args.search_limit <= 0 else args.search_limit,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
