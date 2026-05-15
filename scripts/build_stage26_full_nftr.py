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

from font_machine_learn.stage26_full_nftr import export_full_stage26_nftr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a full-map NFTR from a character list using Stage26 layer assignment.")
    parser.add_argument("--source-nftr", type=Path, default=Path("a.NFTR"))
    parser.add_argument("--map-file", type=Path, default=Path("ds_nftr/a.txt"))
    parser.add_argument("--target-metadata", type=Path, default=Path("data/processed/glyphs/stage1_target/target_metadata.json"))
    parser.add_argument("--font", type=Path, default=Path("fonts/WenQuanYi.Bitmap.Song.13px.ttf"))
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26-fullmap.NFTR"),
    )
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--preview-png", type=Path, default=None)
    parser.add_argument("--start-code", type=lambda value: int(value, 0), default=0xE800)
    parser.add_argument("--core-threshold", type=float, default=0.55)
    parser.add_argument("--shadow-threshold", type=float, default=0.45)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--preview-columns", type=int, default=32)
    parser.add_argument("--preview-scale", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_full_stage26_nftr(
        source_nftr=args.source_nftr,
        map_file=args.map_file,
        target_metadata=args.target_metadata,
        font_path=args.font,
        font_index=args.font_index,
        font_size=args.font_size,
        out_nftr=args.out,
        metadata_json=args.metadata_json,
        preview_png=args.preview_png,
        start_code=args.start_code,
        core_threshold=args.core_threshold,
        shadow_threshold=args.shadow_threshold,
        max_train_glyphs=args.max_train_glyphs,
        preview_columns=args.preview_columns,
        preview_scale=args.preview_scale,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
