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
    SONG13_LAYER_MLP_CONTACT,
    SONG13_LAYER_MLP_ERROR_CONTACT,
    SONG13_LAYER_MLP_METADATA,
    STAGE26_SONG13_LAYER_MLP,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA  # noqa: E402
from font_machine_learn.song13_layer_mlp import (  # noqa: E402
    export_song13_layer_mlp,
    export_target_quantized_source_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage26 source-locked Song13 learned layer assignment.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SONG13_SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=STAGE26_SONG13_LAYER_MLP)
    parser.add_argument("--metadata", type=Path, default=SONG13_LAYER_MLP_METADATA)
    parser.add_argument("--contact-sheet", type=Path, default=SONG13_LAYER_MLP_CONTACT)
    parser.add_argument("--error-contact-sheet", type=Path, default=SONG13_LAYER_MLP_ERROR_CONTACT)
    parser.add_argument("--core-threshold", type=float, action="append", default=[])
    parser.add_argument("--shadow-threshold", type=float, action="append", default=[])
    parser.add_argument("--patch-radius", type=int, default=4)
    parser.add_argument("--max-train-glyphs", type=int, default=None)
    parser.add_argument("--search-limit", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--eval-source-jobs", type=int, default=1)
    parser.add_argument(
        "--eval-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluate an additional source metadata file with the same trained models.",
    )
    parser.add_argument(
        "--eval-target-quantized",
        action="store_true",
        help="Also evaluate target-derived ge2 and eq3 source masks.",
    )
    parser.add_argument("--core-hidden-units", type=int, default=64)
    parser.add_argument("--shadow-hidden-units", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=26)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extra_eval_sources: dict[str, Path] = {}
    for item in args.eval_source:
        if "=" not in item:
            raise SystemExit(f"--eval-source must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        extra_eval_sources[name] = Path(path)
    if args.eval_target_quantized:
        quantized_root = args.out_dir / "eval_sources" / "_target_quantized_sources"
        extra_eval_sources["target_ge2"] = export_target_quantized_source_metadata(
            args.target_metadata,
            out_dir=quantized_root,
            mode="ge2",
        )
        extra_eval_sources["target_eq3"] = export_target_quantized_source_metadata(
            args.target_metadata,
            out_dir=quantized_root,
            mode="eq3",
        )
    result = export_song13_layer_mlp(
        target_metadata=args.target_metadata,
        source_metadata=args.source_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        contact_sheet=args.contact_sheet,
        error_contact_sheet=args.error_contact_sheet,
        core_thresholds=args.core_threshold or None,
        shadow_thresholds=args.shadow_threshold or None,
        patch_radius=args.patch_radius,
        max_train_glyphs=args.max_train_glyphs,
        search_limit=args.search_limit,
        jobs=args.jobs,
        extra_eval_sources=extra_eval_sources or None,
        eval_source_jobs=args.eval_source_jobs,
        core_hidden_units=args.core_hidden_units,
        shadow_hidden_units=args.shadow_hidden_units,
        max_iter=args.max_iter,
        random_seed=args.random_seed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
