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

from font_machine_learn.trainable_baseline import export_mlp_baseline
from font_machine_learn.paths import BASELINE_MLP_CONTACT, BASELINE_MLP_DIR, BASELINE_MLP_METADATA, SOURCE_METADATA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the small pixel-level MLP baseline.")
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=BASELINE_MLP_DIR)
    parser.add_argument("--metadata", type=Path, default=BASELINE_MLP_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=BASELINE_MLP_CONTACT)
    parser.add_argument("--max-train-glyphs", type=int, default=512)
    parser.add_argument("--hidden-units", type=int, default=48)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--columns", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_mlp_baseline(
        source_metadata=args.source_metadata,
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
