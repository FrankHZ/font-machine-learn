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

from font_machine_learn.external_eval import export_external_eval
from font_machine_learn.paths import EXTERNAL_EVAL_METADATA, STAGE21_EXTERNAL_EVAL, TARGET_METADATA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Stage20 two-head patch model on external source masks.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE21_EXTERNAL_EVAL)
    parser.add_argument("--metadata", type=Path, default=EXTERNAL_EVAL_METADATA)
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--core-hidden-units", type=int, default=64)
    parser.add_argument("--shadow-hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=21)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_external_eval(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        patch_radius=args.patch_radius,
        max_train_glyphs=args.max_train_glyphs,
        core_hidden_units=args.core_hidden_units,
        shadow_hidden_units=args.shadow_hidden_units,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
