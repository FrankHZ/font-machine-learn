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

from font_machine_learn.paths import STAGE28_TARGET_QUANTIZED, TARGET_METADATA, TARGET_QUANTIZED_METADATA
from font_machine_learn.target_quantized_calibration import MODES, export_target_quantized_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate 1bpp target quantization before larger models.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE28_TARGET_QUANTIZED)
    parser.add_argument("--metadata", type=Path, default=TARGET_QUANTIZED_METADATA)
    parser.add_argument("--mode", action="append", choices=MODES, default=[])
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--core-threshold", type=float, default=0.55)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--core-hidden-units", type=int, default=64)
    parser.add_argument("--shadow-hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--contact-count", type=int, default=160)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_target_quantized_calibration(
        target_metadata=args.target_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        modes=tuple(args.mode) or MODES,
        patch_radius=args.patch_radius,
        core_threshold=args.core_threshold,
        shadow_threshold=args.shadow_threshold,
        max_train_glyphs=args.max_train_glyphs,
        core_hidden_units=args.core_hidden_units,
        shadow_hidden_units=args.shadow_hidden_units,
        max_iter=args.max_iter,
        contact_count=args.contact_count,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
