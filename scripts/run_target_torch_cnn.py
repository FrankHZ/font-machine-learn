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
    STAGE30_TARGET_TORCH,
    TARGET_METADATA,
    TARGET_TORCH_CONTACT,
    TARGET_TORCH_ERROR_CONTACT,
    TARGET_TORCH_METADATA,
)
from font_machine_learn.target_torch_cnn import export_target_torch_cnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage30 tiny PyTorch CNN target ge2 calibration.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE30_TARGET_TORCH)
    parser.add_argument("--metadata", type=Path, default=TARGET_TORCH_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=TARGET_TORCH_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=TARGET_TORCH_ERROR_CONTACT)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_target_torch_cnn(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
