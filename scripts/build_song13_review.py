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

from font_machine_learn.paths import (  # noqa: E402
    SONG13_ADD_ONLY_METADATA,
    SONG13_LAYER_MLP_METADATA,
    SONG13_REVIEW_METADATA,
    SONG13_REVIEW_OVERVIEW_CONTACT,
    SONG13_SOURCE_LOCKED_METADATA,
    STAGE27_SONG13_REVIEW,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA  # noqa: E402
from font_machine_learn.song13_review import export_song13_review  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage27 Song13 human-review contact sheets.")
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SONG13_SOURCE_METADATA)
    parser.add_argument("--stage24-metadata", type=Path, default=SONG13_ADD_ONLY_METADATA)
    parser.add_argument("--stage25-metadata", type=Path, default=SONG13_SOURCE_LOCKED_METADATA)
    parser.add_argument("--stage26-metadata", type=Path, default=SONG13_LAYER_MLP_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE27_SONG13_REVIEW)
    parser.add_argument("--metadata", type=Path, default=SONG13_REVIEW_METADATA)
    parser.add_argument("--overview-contact-sheet", type=Path, default=SONG13_REVIEW_OVERVIEW_CONTACT)
    parser.add_argument("--category-count", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_song13_review(
        source_metadata=args.source_metadata,
        stage24_metadata=args.stage24_metadata,
        stage25_metadata=args.stage25_metadata,
        stage26_metadata=args.stage26_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        overview_contact_sheet=args.overview_contact_sheet,
        category_count=args.category_count,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
