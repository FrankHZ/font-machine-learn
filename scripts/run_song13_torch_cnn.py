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

from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA  # noqa: E402
from font_machine_learn.song13_torch_cnn import export_song13_torch_cnn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage31: transfer target-trained tiny torch CNN to Song13 masks.")
    parser.add_argument("--target-metadata", type=Path, default=Path("data/processed/glyphs/stage1_target/target_metadata.json"))
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SONG13_SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/glyphs/stage31_song13_torch_cnn"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--error-contact-sheet", type=Path, default=None)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--random-seed", type=int, default=31)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    result = export_song13_torch_cnn(
        target_metadata=args.target_metadata,
        source_metadata=args.source_metadata,
        out_dir=out_dir,
        metadata_json=args.metadata or out_dir / "song13_torch_cnn_metadata.json",
        contact_sheet=args.contact_sheet or out_dir / "song13_torch_cnn_contact.png",
        error_contact_sheet=args.error_contact_sheet or out_dir / "song13_torch_cnn_errors.png",
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        eval_limit=args.eval_limit,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
