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

from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.paths import SOURCE_DIR, SOURCE_METADATA, SOURCE_TARGET_CONTACT, TARGET_METADATA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render WQY source glyphs paired with target NFTR glyphs.")
    parser.add_argument("--nftr", type=Path, default=Path("a.NFTR"))
    parser.add_argument("--font", type=Path, default=Path("wqy-zenhei.ttc"))
    parser.add_argument("--font-index", type=int, default=2)
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=SOURCE_TARGET_CONTACT)
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--font-mode", choices=("L", "1"), default="L", help="Pillow text rasterization mode.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_source_dataset(
        nftr_source=args.nftr,
        font_path=args.font,
        out_dir=args.out_dir,
        target_metadata=args.target_metadata,
        font_index=args.font_index,
        font_size=args.font_size,
        threshold=args.threshold,
        font_mode=args.font_mode,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
