from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.nftr import checkerboard  # noqa: E402


DEFAULT_OUT = Path("docs/assets/stage32_public_comparison.png")
DEFAULT_METADATA = Path("docs/assets/stage32_public_comparison.json")


def load_stage25(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in payload["glyphs"] if row.get("char_class") == "cjk"}


def load_stage26(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload["best_candidate"]
    rows = payload["candidates"][candidate]["glyphs"]
    return {int(row["index"]): row for row in rows if row.get("char_class") == "cjk"}


def load_stage31(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in payload["glyphs"] if row.get("char_class") == "cjk"}


def first_complete_indices(*maps: dict[int, dict], count: int) -> list[int]:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    return sorted(common)[:count]


def paste_scaled(sheet: Image.Image, image_path: str, xy: tuple[int, int], scale: int) -> None:
    image = Image.open(image_path).convert("RGBA")
    scaled = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    sheet.alpha_composite(scaled, xy)


def build_public_comparison(
    *,
    stage25_metadata: Path,
    stage26_metadata: Path,
    stage31_metadata: Path,
    out_png: Path,
    out_json: Path,
    glyph_count: int,
    columns: int,
    scale: int,
    pad: int,
    label_width: int,
) -> None:
    stage25 = load_stage25(stage25_metadata)
    stage26 = load_stage26(stage26_metadata)
    stage31 = load_stage31(stage31_metadata)
    indices = first_complete_indices(stage25, stage26, stage31, count=glyph_count)
    if not indices:
        raise ValueError("No shared CJK glyphs found across Stage25, Stage26, and Stage31 metadata")

    first = Image.open(stage25[indices[0]]["predicted_png"]).convert("RGBA")
    tile_w = first.width * scale
    tile_h = first.height * scale
    rows_per_group = 4
    group_h = rows_per_group * tile_h + (rows_per_group - 1) * pad
    groups_y = (len(indices) + columns - 1) // columns
    width = label_width + columns * tile_w + (columns + 1) * pad
    height = groups_y * group_h + (groups_y + 1) * pad
    sheet = checkerboard((width, height), max(2, scale * 2))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    labels = [
        ("source", "original_source_png"),
        ("stage25", "predicted_png"),
        ("stage26", "predicted_png"),
        ("stage32", "predicted_png"),
    ]

    for position, index in enumerate(indices):
        col = position % columns
        group = position // columns
        x = label_width + pad + col * (tile_w + pad)
        y = pad + group * (group_h + pad)
        records = {
            "source": stage25[index],
            "stage25": stage25[index],
            "stage26": stage26[index],
            "stage32": stage31[index],
        }
        for row, (label, key) in enumerate(labels):
            row_y = y + row * (tile_h + pad)
            if col == 0:
                draw.text((pad, row_y + max(0, (tile_h - 8) // 2)), label, fill=(0, 0, 0, 255), font=font)
            paste_scaled(sheet, records[label][key], (x, row_y), scale)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    payload = {
        "task": "Stage32 public comparison contact sheet",
        "image": str(out_png),
        "glyph_count": len(indices),
        "columns": columns,
        "scale": scale,
        "rows": ["source", "stage25", "stage26", "stage32"],
        "source_note": "No target NFTR glyph row is included in this public README asset.",
        "stage32_note": "The Stage32 comparison row uses the Stage31 Song13 torch-transfer output as the final candidate row.",
        "stage25_metadata": str(stage25_metadata),
        "stage26_metadata": str(stage26_metadata),
        "stage31_metadata": str(stage31_metadata),
        "indices": indices,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage32 public comparison contact sheet.")
    parser.add_argument("--stage25-metadata", type=Path, default=Path("data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_metadata.json"))
    parser.add_argument("--stage26-metadata", type=Path, default=Path("data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_metadata.json"))
    parser.add_argument("--stage31-metadata", type=Path, default=Path("data/processed/glyphs/stage31_song13_torch_cnn/song13_torch_cnn_metadata.json"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--glyph-count", type=int, default=48)
    parser.add_argument("--columns", type=int, default=24)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--label-width", type=int, default=42)
    args = parser.parse_args()
    build_public_comparison(
        stage25_metadata=args.stage25_metadata,
        stage26_metadata=args.stage26_metadata,
        stage31_metadata=args.stage31_metadata,
        out_png=args.out,
        out_json=args.metadata,
        glyph_count=args.glyph_count,
        columns=args.columns,
        scale=args.scale,
        pad=args.pad,
        label_width=args.label_width,
    )
    print(json.dumps({"out": str(args.out), "metadata": str(args.metadata)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
