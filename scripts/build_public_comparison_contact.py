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
from font_machine_learn.source_font import quantize_mask_to_1bpp, render_mask  # noqa: E402


DEFAULT_OUT_DIR = Path("docs/assets")


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


def first_char(chars: list[str]) -> str:
    for char in chars:
        if char:
            return char
    return " "


def render_source_image(
    char: str,
    *,
    font: ImageFont.FreeTypeFont,
    cell_width: int,
    cell_height: int,
    threshold: int,
    font_mode: str,
    x_offset: int,
    y_offset: int,
) -> Image.Image:
    mask, _bbox = render_mask(
        font,
        char,
        cell_width,
        cell_height,
        font_mode=font_mode,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    return quantize_mask_to_1bpp(mask, threshold)


def paste_image_scaled(sheet: Image.Image, image: Image.Image, xy: tuple[int, int], scale: int) -> None:
    scaled = image.convert("RGBA").resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    sheet.alpha_composite(scaled, xy)


def paste_path_scaled(sheet: Image.Image, image_path: str, xy: tuple[int, int], scale: int) -> None:
    paste_image_scaled(sheet, Image.open(image_path).convert("RGBA"), xy, scale)


def make_sheet(
    *,
    name: str,
    font: ImageFont.FreeTypeFont,
    source_config: dict,
    indices: list[int],
    stage25: dict[int, dict],
    stage26: dict[int, dict],
    stage32: dict[int, dict],
    out_png: Path,
    glyph_count: int,
    columns: int,
    scale: int,
    pad: int,
    label_width: int,
) -> dict:
    selected = indices[:glyph_count]
    first = Image.open(stage25[selected[0]]["predicted_png"]).convert("RGBA")
    cell_width = first.width
    cell_height = first.height
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    labels = ["source", "stage25", "stage26", "stage32"]
    group_h = len(labels) * tile_h + (len(labels) - 1) * pad
    groups_y = (len(selected) + columns - 1) // columns
    width = label_width + columns * tile_w + (columns + 1) * pad
    height = groups_y * group_h + (groups_y + 1) * pad
    sheet = checkerboard((width, height), max(2, scale * 2))
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()

    for position, index in enumerate(selected):
        col = position % columns
        group = position // columns
        x = label_width + pad + col * (tile_w + pad)
        y = pad + group * (group_h + pad)
        char = first_char(list(stage25[index]["chars"]))
        row_images = {
            "source": render_source_image(
                char,
                font=font,
                cell_width=cell_width,
                cell_height=cell_height,
                **source_config,
            ),
            "stage25": stage25[index]["predicted_png"],
            "stage26": stage26[index]["predicted_png"],
            "stage32": stage32[index]["predicted_png"],
        }
        for row, label in enumerate(labels):
            row_y = y + row * (tile_h + pad)
            if col == 0:
                draw.text((pad, row_y + max(0, (tile_h - 8) // 2)), label, fill=(0, 0, 0, 255), font=label_font)
            image_or_path = row_images[label]
            if isinstance(image_or_path, Image.Image):
                paste_image_scaled(sheet, image_or_path, (x, row_y), scale)
            else:
                paste_path_scaled(sheet, image_or_path, (x, row_y), scale)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return {
        "name": name,
        "image": str(out_png),
        "rows": labels,
        "glyph_count": len(selected),
        "indices": selected,
    }


def build_public_comparison(
    *,
    stage25_metadata: Path,
    stage26_metadata: Path,
    stage31_metadata: Path,
    out_dir: Path,
    metadata_json: Path,
    glyph_count: int,
    columns: int,
    scale: int,
    pad: int,
    label_width: int,
) -> None:
    stage25 = load_stage25(stage25_metadata)
    stage26 = load_stage26(stage26_metadata)
    stage32 = load_stage31(stage31_metadata)
    indices = first_complete_indices(stage25, stage26, stage32, count=glyph_count)
    if not indices:
        raise ValueError("No shared CJK glyphs found across Stage25, Stage26, and Stage31 metadata")

    source_config = {"threshold": 96, "font_mode": "L", "x_offset": -1, "y_offset": 1}
    sheets = [
        make_sheet(
            name="song13",
            font=ImageFont.truetype("fonts/WenQuanYi.Bitmap.Song.13px.ttf", size=15, index=0),
            source_config=source_config,
            indices=indices,
            stage25=stage25,
            stage26=stage26,
            stage32=stage32,
            out_png=out_dir / "stage32_public_comparison_song13.png",
            glyph_count=glyph_count,
            columns=columns,
            scale=scale,
            pad=pad,
            label_width=label_width,
        ),
        make_sheet(
            name="song12",
            font=ImageFont.truetype("fonts/WenQuanYi.Bitmap.Song.12px.ttf", size=15, index=0),
            source_config=source_config,
            indices=indices,
            stage25=stage25,
            stage26=stage26,
            stage32=stage32,
            out_png=out_dir / "stage32_public_comparison_song12.png",
            glyph_count=glyph_count,
            columns=columns,
            scale=scale,
            pad=pad,
            label_width=label_width,
        ),
    ]
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(
        json.dumps(
            {
                "task": "Stage32 public comparison contact sheets",
                "rows": ["source", "stage25", "stage26", "stage32"],
                "source_note": "No target NFTR glyph row is included in public README assets.",
                "stage32_note": "The Stage32 row uses the Stage31 Song13 torch-transfer output.",
                "sheets": sheets,
                "stage25_metadata": str(stage25_metadata),
                "stage26_metadata": str(stage26_metadata),
                "stage31_metadata": str(stage31_metadata),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage32 public comparison contact sheets.")
    parser.add_argument("--stage25-metadata", type=Path, default=Path("data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_metadata.json"))
    parser.add_argument("--stage26-metadata", type=Path, default=Path("data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_metadata.json"))
    parser.add_argument("--stage31-metadata", type=Path, default=Path("data/processed/glyphs/stage31_song13_torch_cnn/song13_torch_cnn_metadata.json"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_OUT_DIR / "stage32_public_comparison.json")
    parser.add_argument("--glyph-count", type=int, default=48)
    parser.add_argument("--columns", type=int, default=24)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--label-width", type=int, default=50)
    args = parser.parse_args()
    build_public_comparison(
        stage25_metadata=args.stage25_metadata,
        stage26_metadata=args.stage26_metadata,
        stage31_metadata=args.stage31_metadata,
        out_dir=args.out_dir,
        metadata_json=args.metadata,
        glyph_count=args.glyph_count,
        columns=args.columns,
        scale=args.scale,
        pad=args.pad,
        label_width=args.label_width,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "metadata": str(args.metadata)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
