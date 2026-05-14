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
    TARGET_EQ3_1BPP_DIR,
    TARGET_GE2_1BPP_DIR,
    TARGET_MASK_COMPARE_CONTACT,
    TARGET_MASK_COMPARE_METADATA,
)
from font_machine_learn.target_mask_compare import export_target_mask_compare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare target-derived >=2 and ==3 masks against WQY source.")
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--ge2-dir", type=Path, default=TARGET_GE2_1BPP_DIR)
    parser.add_argument("--eq3-dir", type=Path, default=TARGET_EQ3_1BPP_DIR)
    parser.add_argument("--metadata", type=Path, default=TARGET_MASK_COMPARE_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=TARGET_MASK_COMPARE_CONTACT)
    parser.add_argument("--contact-count", type=int, default=192)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_target_mask_compare(
        source_metadata=args.source_metadata,
        ge2_dir=args.ge2_dir,
        eq3_dir=args.eq3_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        contact_count=args.contact_count,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
