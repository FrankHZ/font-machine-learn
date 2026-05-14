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

from font_machine_learn.paths import STYLE_MLP_CONTACT, STYLE_MLP_METADATA, STAGE15_STYLE_MLP, TARGET_METADATA
from font_machine_learn.style_mlp import export_style_mlp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train controlled 1bpp-to-2bpp style MLP baselines.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE15_STYLE_MLP)
    parser.add_argument("--metadata", type=Path, default=STYLE_MLP_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=STYLE_MLP_CONTACT)
    parser.add_argument("--max-train-glyphs", type=int, default=768)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=15)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_style_mlp(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        max_train_glyphs=args.max_train_glyphs,
        hidden_units=args.hidden_units,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
        scale=args.scale,
        columns=args.columns,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
