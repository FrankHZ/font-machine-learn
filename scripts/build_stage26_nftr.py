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

from font_machine_learn.stage26_nftr import export_stage26_nftr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an original-layout NFTR using Stage26 predicted glyph layers.")
    parser.add_argument("--source-nftr", type=Path, default=Path("a.NFTR"))
    parser.add_argument(
        "--stage26-metadata",
        type=Path,
        default=Path("data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_metadata.json"),
    )
    parser.add_argument("--candidate", default=None, help="Stage26 candidate name. Default: metadata best_candidate.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26.NFTR"),
    )
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--preview-png", type=Path, default=None)
    parser.add_argument("--preview-columns", type=int, default=32)
    parser.add_argument("--preview-scale", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_stage26_nftr(
        source_nftr=args.source_nftr,
        stage26_metadata=args.stage26_metadata,
        candidate=args.candidate,
        out_nftr=args.out,
        metadata_json=args.metadata_json,
        preview_png=args.preview_png,
        preview_columns=args.preview_columns,
        preview_scale=args.preview_scale,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
