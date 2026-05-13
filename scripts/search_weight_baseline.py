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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search source glyph weight/offset before shadowing.")
    parser.add_argument("--source-metadata", type=Path, default=Path("data/processed/glyphs/source_metadata.json"))
    parser.add_argument("--source-out-dir", type=Path, default=Path("data/processed/glyphs/source_weighted"))
    parser.add_argument("--shadow-out-dir", type=Path, default=Path("data/processed/glyphs/baseline_weighted_shadow"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/glyphs/weight_search_metadata.json"))
    parser.add_argument("--search-json", type=Path, default=Path("data/processed/glyphs/weight_search.json"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("data/processed/glyphs/baseline_weighted_shadow_contact.png"))
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
