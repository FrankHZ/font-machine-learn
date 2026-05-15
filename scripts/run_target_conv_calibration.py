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

from font_machine_learn.paths import TARGET_CONV_CONTACT, TARGET_CONV_ERROR_CONTACT, TARGET_CONV_METADATA, STAGE29_TARGET_CONV, TARGET_METADATA
from font_machine_learn.target_conv_calibration import export_target_conv_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage29 lightweight convolution-feature target ge2 calibration.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE29_TARGET_CONV)
    parser.add_argument("--metadata", type=Path, default=TARGET_CONV_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=TARGET_CONV_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=TARGET_CONV_ERROR_CONTACT)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_target_conv_calibration(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        max_train_glyphs=args.max_train_glyphs,
        hidden_units=args.hidden_units,
        max_iter=args.max_iter,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
