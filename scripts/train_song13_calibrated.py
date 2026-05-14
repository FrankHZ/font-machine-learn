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
    SONG13_CALIBRATED_CONTACT,
    SONG13_CALIBRATED_ERROR_CONTACT,
    SONG13_CALIBRATED_METADATA,
    STAGE23_SONG13_CALIBRATED,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA  # noqa: E402
from font_machine_learn.song13_calibrated import export_song13_calibrated  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage23 calibrated Song13 adapter threshold sweep.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SONG13_SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE23_SONG13_CALIBRATED)
    parser.add_argument("--metadata", type=Path, default=SONG13_CALIBRATED_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=SONG13_CALIBRATED_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=SONG13_CALIBRATED_ERROR_CONTACT)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--adapter-patch-radius", type=int, default=3)
    parser.add_argument("--style-patch-radius", type=int, default=4)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--adapter-hidden-units", type=int, default=64)
    parser.add_argument("--style-core-hidden-units", type=int, default=64)
    parser.add_argument("--style-shadow-hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=23)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_song13_calibrated(
        target_metadata=args.target_metadata,
        source_metadata=args.source_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        thresholds=args.threshold or None,
        adapter_patch_radius=args.adapter_patch_radius,
        style_patch_radius=args.style_patch_radius,
        max_train_glyphs=args.max_train_glyphs,
        adapter_hidden_units=args.adapter_hidden_units,
        style_core_hidden_units=args.style_core_hidden_units,
        style_shadow_hidden_units=args.style_shadow_hidden_units,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
