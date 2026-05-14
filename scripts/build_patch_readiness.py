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

from font_machine_learn.patch_readiness import export_patch_readiness
from font_machine_learn.paths import PATCH_READINESS_CONTACT, PATCH_READINESS_METADATA, PATCH_READINESS_PATCHES, TARGET_METADATA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 18 patch-readiness diagnostics.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--metadata", type=Path, default=PATCH_READINESS_METADATA)
    parser.add_argument("--patches-jsonl", type=Path, default=PATCH_READINESS_PATCHES)
    parser.add_argument("--contact-sheet", type=Path, default=PATCH_READINESS_CONTACT)
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--examples-per-category", type=int, default=48)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_patch_readiness(
        target_metadata=args.target_metadata,
        metadata_json=args.metadata,
        patches_jsonl=args.patches_jsonl,
        contact_sheet=args.contact_sheet,
        patch_radius=args.patch_radius,
        examples_per_category=args.examples_per_category,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
