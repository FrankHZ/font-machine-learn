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

from font_machine_learn.binary_diagnostic import export_binary_diagnostic


def baseline_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline must use NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize target glyphs to 1bpp and score baselines.")
    parser.add_argument("--source-metadata", type=Path, default=Path("data/processed/glyphs/source_metadata.json"))
    parser.add_argument("--baseline", action="append", type=baseline_spec, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/glyphs/target_1bpp"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/glyphs/binary_diagnostic_metadata.json"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("data/processed/glyphs/binary_diagnostic_contact.png"))
    parser.add_argument("--worst-count", type=int, default=160)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baselines = dict(args.baseline) if args.baseline else None
    result = export_binary_diagnostic(
        source_metadata=args.source_metadata,
        baseline_metadata=baselines,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        worst_count=args.worst_count,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
