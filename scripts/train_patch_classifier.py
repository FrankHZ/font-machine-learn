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

from font_machine_learn.patch_classifier import export_patch_classifier
from font_machine_learn.paths import (
    PATCH_CLASSIFIER_CONTACT,
    PATCH_CLASSIFIER_ERROR_CONTACT,
    PATCH_CLASSIFIER_METADATA,
    STAGE19_PATCH_CLASSIFIER,
    TARGET_METADATA,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 19 patch-only ge2 level-2/3 classifiers.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE19_PATCH_CLASSIFIER)
    parser.add_argument("--metadata", type=Path, default=PATCH_CLASSIFIER_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=PATCH_CLASSIFIER_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=PATCH_CLASSIFIER_ERROR_CONTACT)
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=19)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_patch_classifier(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        patch_radius=args.patch_radius,
        max_train_glyphs=args.max_train_glyphs,
        hidden_units=args.hidden_units,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
