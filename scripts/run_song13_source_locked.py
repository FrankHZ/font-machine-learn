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
    SONG13_SOURCE_LOCKED_CONTACT,
    SONG13_SOURCE_LOCKED_ERROR_CONTACT,
    SONG13_SOURCE_LOCKED_METADATA,
    SONG13_SOURCE_LOCKED_SEARCH,
    STAGE25_SONG13_SOURCE_LOCKED,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA  # noqa: E402
from font_machine_learn.song13_source_locked import export_song13_source_locked  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage25 source-locked Song13 style rule sweep.")
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SONG13_SOURCE_METADATA)
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE25_SONG13_SOURCE_LOCKED)
    parser.add_argument("--metadata", type=Path, default=SONG13_SOURCE_LOCKED_METADATA)
    parser.add_argument("--search-json", type=Path, default=SONG13_SOURCE_LOCKED_SEARCH)
    parser.add_argument("--contact-sheet", type=Path, default=SONG13_SOURCE_LOCKED_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=SONG13_SOURCE_LOCKED_ERROR_CONTACT)
    parser.add_argument("--search-limit", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_song13_source_locked(
        source_metadata=args.source_metadata,
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        search_json=args.search_json,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        search_limit=args.search_limit,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
