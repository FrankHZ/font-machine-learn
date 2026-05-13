from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.nftr import checkerboard, palette_rgba
from font_machine_learn.paths import BASELINE_SHADOW_CONTACT, BASELINE_SHADOW_DIR, BASELINE_SHADOW_METADATA, SOURCE_METADATA
from font_machine_learn.source_font import export_source_dataset


@dataclass(frozen=True)
class BaselineGlyphMetrics:
    index: int
    codes: list[int]
    chars: list[str]
    source_png: str
    predicted_png: str
    target_png: str
    pixel_accuracy: float
    mean_absolute_error: float
    foreground_iou: float


@dataclass(frozen=True)
class BaselineExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    mean_pixel_accuracy: float
    mean_absolute_error: float
    mean_foreground_iou: float


def source_image_to_levels(source: Image.Image) -> list[list[int]]:
    alpha = source.convert("RGBA").getchannel("A")
    width, height = alpha.size
    levels = [[0 for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            if alpha.getpixel((x, y)):
                levels[y][x] = 3
    return levels


def add_shadow_levels(levels: list[list[int]]) -> list[list[int]]:
    height = len(levels)
    width = len(levels[0]) if height else 0
    out = [row[:] for row in levels]
    for y in range(height):
        for x in range(width):
            if levels[y][x] != 3:
                continue
            for dx, dy, value in ((1, 1, 1), (1, 0, 2), (0, 1, 2)):
                sx = x + dx
                sy = y + dy
                if 0 <= sx < width and 0 <= sy < height and out[sy][sx] == 0:
                    out[sy][sx] = value
    return out


def levels_to_image(levels: list[list[int]]) -> Image.Image:
    height = len(levels)
    width = len(levels[0]) if height else 0
    image = Image.new("RGBA", (width, height))
    image.putdata([palette_rgba(value) for row in levels for value in row])
    return image


def image_to_target_levels(image: Image.Image) -> list[list[int]]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    levels = [[0 for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            r, _g, _b, a = rgba.getpixel((x, y))
            if a == 0:
                value = 0
            elif r <= 32:
                value = 3
            elif r <= 128:
                value = 2
            else:
                value = 1
            levels[y][x] = value
    return levels


def flatten(levels: list[list[int]]) -> list[int]:
    return [value for row in levels for value in row]


def compare_levels(predicted: list[list[int]], target: list[list[int]]) -> tuple[float, float, float]:
    pred = flatten(predicted)
    actual = flatten(target)
    if len(pred) != len(actual):
        raise ValueError("predicted and target level grids have different sizes")

    count = len(actual)
    matches = sum(1 for left, right in zip(pred, actual) if left == right)
    mae = sum(abs(left - right) for left, right in zip(pred, actual)) / count
    pred_fg = {i for i, value in enumerate(pred) if value > 0}
    actual_fg = {i for i, value in enumerate(actual) if value > 0}
    union = pred_fg | actual_fg
    iou = 1.0 if not union else len(pred_fg & actual_fg) / len(union)
    return matches / count, mae, iou


def make_baseline_contact_sheet(
    records: list[BaselineGlyphMetrics],
    *,
    scale: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    pad: int,
) -> Image.Image:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    group_h = tile_h * 3 + pad * 2
    rows = (len(records) + columns - 1) // columns
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, rows * group_h + (rows + 1) * pad),
        max(2, scale * 2),
    )

    for record in records:
        x = pad + (record.index % columns) * (tile_w + pad)
        y = pad + (record.index // columns) * (group_h + pad)
        images = (
            Image.open(record.source_png).convert("RGBA"),
            Image.open(record.predicted_png).convert("RGBA"),
            Image.open(record.target_png).convert("RGBA"),
        )
        for row, image in enumerate(images):
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + row * (tile_h + pad)))
    return sheet


def export_shadow_baseline(
    source_metadata: Path = SOURCE_METADATA,
    out_dir: Path = BASELINE_SHADOW_DIR,
    *,
    metadata_json: Path | None = BASELINE_SHADOW_METADATA,
    contact_sheet: Path | None = BASELINE_SHADOW_CONTACT,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> BaselineExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_json or out_dir.parent / "baseline_shadow_metadata.json"
    contact_path = contact_sheet or out_dir.parent / "baseline_shadow_contact.png"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[BaselineGlyphMetrics] = []
    for glyph in source["glyphs"]:
        index = int(glyph["index"])
        source_png = str(glyph["source_png"])
        target_png = str(glyph["target_png"])
        source_image = Image.open(source_png).convert("RGBA")
        target_image = Image.open(target_png).convert("RGBA")

        predicted_levels = add_shadow_levels(source_image_to_levels(source_image))
        target_levels = image_to_target_levels(target_image)
        predicted_image = levels_to_image(predicted_levels)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        predicted_image.save(predicted_png)
        accuracy, mae, iou = compare_levels(predicted_levels, target_levels)

        records.append(
            BaselineGlyphMetrics(
                index=index,
                codes=list(glyph["codes"]),
                chars=list(glyph["chars"]),
                source_png=source_png,
                predicted_png=str(predicted_png),
                target_png=target_png,
                pixel_accuracy=accuracy,
                mean_absolute_error=mae,
                foreground_iou=iou,
            )
        )

    contact = make_baseline_contact_sheet(
        records,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    contact.save(contact_path)

    glyph_count = len(records)
    mean_accuracy = sum(record.pixel_accuracy for record in records) / glyph_count
    mean_mae = sum(record.mean_absolute_error for record in records) / glyph_count
    mean_iou = sum(record.foreground_iou for record in records) / glyph_count
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": glyph_count,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "rule": "source level 3 plus right/down level 2 and down-right level 1 shadow",
        "mean_pixel_accuracy": mean_accuracy,
        "mean_absolute_error": mean_mae,
        "mean_foreground_iou": mean_iou,
        "glyph_dir": str(out_dir),
        "contact_sheet": str(contact_path),
        "glyphs": [asdict(record) for record in records],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return BaselineExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_path),
        contact_sheet=str(contact_path),
        glyph_count=glyph_count,
        mean_pixel_accuracy=mean_accuracy,
        mean_absolute_error=mean_mae,
        mean_foreground_iou=mean_iou,
    )
