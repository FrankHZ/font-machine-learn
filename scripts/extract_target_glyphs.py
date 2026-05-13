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

from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import TARGET_CONTACT, TARGET_DIR, TARGET_METADATA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a reversed-tag NFTR into target glyph PNGs.")
    parser.add_argument("source", type=Path, default=Path("a.NFTR"), nargs="?")
    parser.add_argument("--out-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=TARGET_CONTACT)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_target_dataset(
        args.source,
        args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
