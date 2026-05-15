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

from font_machine_learn.baseline import levels_to_image  # noqa: E402
from font_machine_learn.nftr import checkerboard  # noqa: E402
from font_machine_learn.paths import TARGET_METADATA  # noqa: E402
from font_machine_learn.song13_layer_mlp import predict_source_locked_levels  # noqa: E402
from font_machine_learn.song13_source_locked import SourceLockedRule, source_locked_levels  # noqa: E402
from font_machine_learn.song13_torch_cnn import train_target_ge2_cnn  # noqa: E402
from font_machine_learn.source_font import quantize_mask_to_1bpp, render_mask  # noqa: E402
from font_machine_learn.stage26_full_nftr import train_stage26_heads  # noqa: E402
from font_machine_learn.target_torch_cnn import predict_levels  # noqa: E402


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


def source_image_to_mask(image: Image.Image) -> list[list[bool]]:
    alpha = image.convert("RGBA").getchannel("A")
    width, height = alpha.size
    return [[alpha.getpixel((x, y)) > 0 for x in range(width)] for y in range(height)]


def mask_to_tensor(mask: list[list[bool]]):
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(mask, dtype=np.float32)[None, :, :])


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
    stage26_models: tuple[object, object],
    stage32_model: object,
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
        source_image = render_source_image(
            char,
            font=font,
            cell_width=cell_width,
            cell_height=cell_height,
            **source_config,
        )
        source_mask = source_image_to_mask(source_image)
        stage25_levels = source_locked_levels(
            source_mask,
            SourceLockedRule("edge_n1_diag_plus_right_from_source", 1, "diag_plus_right", "source"),
        )
        stage26_levels = predict_source_locked_levels(
            stage26_models[0],
            stage26_models[1],
            source_mask,
            patch_radius=4,
            core_threshold=0.55,
            shadow_threshold=0.45,
        )
        stage32_levels = predict_levels(stage32_model, mask_to_tensor(source_mask))
        row_images = {
            "source": source_image,
            "stage25": levels_to_image(stage25_levels),
            "stage26": levels_to_image(stage26_levels),
            "stage32": levels_to_image(stage32_levels),
        }
        for row, label in enumerate(labels):
            row_y = y + row * (tile_h + pad)
            if col == 0:
                draw.text((pad, row_y + max(0, (tile_h - 8) // 2)), label, fill=(0, 0, 0, 255), font=label_font)
            paste_image_scaled(sheet, row_images[label], (x, row_y), scale)

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
    target_metadata: Path,
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
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    stage26_models = train_stage26_heads(
        target_metadata,
        patch_radius=4,
        max_train_glyphs=None,
        core_hidden_units=64,
        shadow_hidden_units=64,
        max_iter=80,
        random_seed=26,
    )
    stage32_model, _losses, _device, _cuda_device = train_target_ge2_cnn(
        list(target["glyphs"]),
        channels=48,
        epochs=200,
        batch_size=128,
        learning_rate=0.003,
        random_seed=32,
    )
    sheets = [
        make_sheet(
            name="song13",
            font=ImageFont.truetype("fonts/WenQuanYi.Bitmap.Song.13px.ttf", size=15, index=0),
            source_config=source_config,
            indices=indices,
            stage25=stage25,
            stage26_models=stage26_models,
            stage32_model=stage32_model,
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
            stage26_models=stage26_models,
            stage32_model=stage32_model,
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
                "target_metadata": str(target_metadata),
                "stage_rows_note": "Stage25, Stage26, and Stage32 rows are generated from each sheet's own source font.",
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
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
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
        target_metadata=args.target_metadata,
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
