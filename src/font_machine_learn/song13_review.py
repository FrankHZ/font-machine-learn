from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.nftr import checkerboard
from font_machine_learn.paths import (
    SONG13_ADD_ONLY_METADATA,
    SONG13_LAYER_MLP_METADATA,
    SONG13_REVIEW_METADATA,
    SONG13_REVIEW_OVERVIEW_CONTACT,
    SONG13_SOURCE_LOCKED_METADATA,
    STAGE27_SONG13_REVIEW,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA, Mask


LevelGrid = list[list[int]]


@dataclass(frozen=True)
class Song13ReviewExport:
    out_dir: str
    metadata_json: str
    overview_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    category_count: int


@dataclass(frozen=True)
class ReviewModel:
    name: str
    metadata_json: Path


def mask_from_png(path: str | Path) -> Mask:
    image = Image.open(path).convert("RGBA")
    width, height = image.size
    alpha = image.getchannel("A")
    return [[alpha.getpixel((x, y)) > 0 for x in range(width)] for y in range(height)]


def foreground_count(mask: Mask) -> int:
    return sum(1 for row in mask for value in row if value)


def level_counts(levels: LevelGrid) -> dict[str, int]:
    counts = {str(value): 0 for value in range(4)}
    for row in levels:
        for value in row:
            counts[str(value)] += 1
    return counts


def hole_mask(source_mask: Mask) -> Mask:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    visited = [[False for _x in range(width)] for _y in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        for y in (0, height - 1):
            if height and not source_mask[y][x] and not visited[y][x]:
                visited[y][x] = True
                queue.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            if width and not source_mask[y][x] and not visited[y][x]:
                visited[y][x] = True
                queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx = x + dx
            sy = y + dy
            if 0 <= sx < width and 0 <= sy < height and not source_mask[sy][sx] and not visited[sy][sx]:
                visited[sy][sx] = True
                queue.append((sx, sy))

    return [
        [not source_mask[y][x] and not visited[y][x] for x in range(width)]
        for y in range(height)
    ]


def values_at(levels: LevelGrid, mask: Mask, predicate) -> int:
    return sum(
        1
        for y, row in enumerate(mask)
        for x, value in enumerate(row)
        if value and predicate(levels[y][x])
    )


def source_deleted_ratio(source_mask: Mask, levels: LevelGrid) -> float:
    source_pixels = foreground_count(source_mask)
    if not source_pixels:
        return 0.0
    deleted = sum(
        1
        for y, row in enumerate(source_mask)
        for x, value in enumerate(row)
        if value and levels[y][x] == 0
    )
    return deleted / source_pixels


def source_mask_to_levels(source_mask: Mask) -> LevelGrid:
    return [[3 if value else 0 for value in row] for row in source_mask]


def summarize_levels(source_mask: Mask, levels: LevelGrid) -> dict[str, float | int | dict[str, int]]:
    source_pixels = foreground_count(source_mask)
    total_pixels = len(source_mask) * len(source_mask[0]) if source_mask else 0
    holes = hole_mask(source_mask)
    hole_pixels = foreground_count(holes)
    counts = level_counts(levels)
    source_level2 = values_at(levels, source_mask, lambda value: value == 2)
    source_level3 = values_at(levels, source_mask, lambda value: value == 3)
    shadow_pixels = counts["1"]
    filled_holes = values_at(levels, holes, lambda value: value > 0)
    return {
        "level_counts": counts,
        "source_pixels": source_pixels,
        "foreground_pixels": total_pixels - counts["0"],
        "source_deleted_ratio": source_deleted_ratio(source_mask, levels),
        "source_level2_ratio": 0.0 if source_pixels == 0 else source_level2 / source_pixels,
        "source_level3_ratio": 0.0 if source_pixels == 0 else source_level3 / source_pixels,
        "shadow_per_source": 0.0 if source_pixels == 0 else shadow_pixels / source_pixels,
        "hole_pixels": hole_pixels,
        "hole_fill_ratio": 0.0 if hole_pixels == 0 else filled_holes / hole_pixels,
    }


def load_stage24_best(path: Path) -> tuple[str, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload["best_candidate"]
    return best, {int(glyph["index"]): glyph for glyph in payload["candidates"][best]["glyphs"]}


def load_stage25_best(path: Path) -> tuple[str, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload["best_rule"]["name"]
    return best, {int(glyph["index"]): glyph for glyph in payload["glyphs"]}


def load_stage26_best(path: Path) -> tuple[str, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload["best_candidate"]
    return best, {int(glyph["index"]): glyph for glyph in payload["candidates"][best]["glyphs"]}


def pixel_disagreement(level_sets: list[LevelGrid]) -> int:
    if not level_sets:
        return 0
    height = len(level_sets[0])
    width = len(level_sets[0][0]) if height else 0
    count = 0
    for y in range(height):
        for x in range(width):
            values = {levels[y][x] for levels in level_sets}
            if len(values) > 1:
                count += 1
    return count


def representative_records(records: list[dict], *, count: int) -> list[dict]:
    if len(records) <= count:
        return records
    step = (len(records) - 1) / max(1, count - 1)
    return [records[round(i * step)] for i in range(count)]


def select_categories(records: list[dict], *, count: int) -> dict[str, list[dict]]:
    cjk = [record for record in records if record["char_class"] == "cjk"]
    by_source = sorted(cjk, key=lambda record: int(record["source_pixels"]))
    by_disagreement = sorted(cjk, key=lambda record: int(record["model_disagreement_pixels"]), reverse=True)
    by_shadow = sorted(cjk, key=lambda record: float(record["models"]["stage25"]["shadow_per_source"]), reverse=True)
    by_level2 = sorted(cjk, key=lambda record: float(record["models"]["stage25"]["source_level2_ratio"]), reverse=True)
    by_holes = sorted(cjk, key=lambda record: float(record["max_hole_fill_ratio"]), reverse=True)
    return {
        "representative_cjk": representative_records(by_source, count=count),
        "simple_cjk": by_source[:count],
        "complex_cjk": by_source[-count:],
        "model_disagreement": by_disagreement[:count],
        "heavy_shadow_stage25": by_shadow[:count],
        "high_gray2_stage25": by_level2[:count],
        "hole_fill_risk": by_holes[:count],
    }


def make_horizontal_contact_sheet(
    records: list[dict],
    *,
    image_keys: list[str],
    out_path: Path,
    cell_width: int,
    cell_height: int,
    scale: int,
    columns: int,
    pad: int,
) -> None:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    group_w = len(image_keys) * tile_w + (len(image_keys) - 1) * pad
    rows = (len(records) + columns - 1) // columns
    sheet = checkerboard(
        (columns * group_w + (columns + 1) * pad, rows * tile_h + (rows + 1) * pad),
        max(2, scale * 2),
    )
    for position, record in enumerate(records):
        x = pad + (position % columns) * (group_w + pad)
        y = pad + (position // columns) * (tile_h + pad)
        for image_index, key in enumerate(image_keys):
            image = Image.open(record[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x + image_index * (tile_w + pad), y))
    sheet.save(out_path)


def export_song13_review(
    *,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    stage24_metadata: Path = SONG13_ADD_ONLY_METADATA,
    stage25_metadata: Path = SONG13_SOURCE_LOCKED_METADATA,
    stage26_metadata: Path = SONG13_LAYER_MLP_METADATA,
    out_dir: Path = STAGE27_SONG13_REVIEW,
    metadata_json: Path = SONG13_REVIEW_METADATA,
    overview_contact_sheet: Path = SONG13_REVIEW_OVERVIEW_CONTACT,
    category_count: int = 64,
    scale: int = 4,
    columns: int = 4,
    pad: int = 1,
) -> Song13ReviewExport:
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    source_by_index = {int(glyph["index"]): glyph for glyph in source["glyphs"]}
    stage24_name, stage24 = load_stage24_best(stage24_metadata)
    stage25_name, stage25 = load_stage25_best(stage25_metadata)
    stage26_name, stage26 = load_stage26_best(stage26_metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    overview_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for index, source_glyph in source_by_index.items():
        if index not in stage24 or index not in stage25 or index not in stage26:
            continue
        source_mask = mask_from_png(source_glyph["source_png"])
        levels24 = image_to_target_levels(Image.open(stage24[index]["predicted_png"]).convert("RGBA"))
        levels25 = image_to_target_levels(Image.open(stage25[index]["predicted_png"]).convert("RGBA"))
        levels26 = image_to_target_levels(Image.open(stage26[index]["predicted_png"]).convert("RGBA"))
        summaries = {
            "stage24": summarize_levels(source_mask, levels24),
            "stage25": summarize_levels(source_mask, levels25),
            "stage26": summarize_levels(source_mask, levels26),
        }
        record = {
            "index": index,
            "codes": list(source_glyph["codes"]),
            "chars": list(source_glyph["chars"]),
            "char_class": classify_glyph(list(source_glyph["chars"])),
            "source_pixels": foreground_count(source_mask),
            "max_hole_fill_ratio": max(float(summary["hole_fill_ratio"]) for summary in summaries.values()),
            "model_disagreement_pixels": pixel_disagreement([levels24, levels25, levels26]),
            "source_png": source_glyph["source_png"],
            "stage24_adapted_png": stage24[index]["source_png"],
            "stage24_predicted_png": stage24[index]["predicted_png"],
            "stage25_predicted_png": stage25[index]["predicted_png"],
            "stage26_predicted_png": stage26[index]["predicted_png"],
            "target_png": source_glyph["target_png"],
            "models": summaries,
        }
        records.append(record)

    categories = select_categories(records, count=category_count)
    image_keys = [
        "source_png",
        "stage24_adapted_png",
        "stage24_predicted_png",
        "stage25_predicted_png",
        "stage26_predicted_png",
        "target_png",
    ]
    category_payload: dict[str, dict] = {}
    for name, selected in categories.items():
        contact = out_dir / f"{name}.png"
        make_horizontal_contact_sheet(
            selected,
            image_keys=image_keys,
            out_path=contact,
            cell_width=cell_width,
            cell_height=cell_height,
            scale=scale,
            columns=columns,
            pad=pad,
        )
        category_payload[name] = {
            "contact_sheet": str(contact),
            "selected_count": len(selected),
            "glyphs": selected,
        }
    make_horizontal_contact_sheet(
        categories["representative_cjk"],
        image_keys=image_keys,
        out_path=overview_contact_sheet,
        cell_width=cell_width,
        cell_height=cell_height,
        scale=scale,
        columns=columns,
        pad=pad,
    )

    cjk_count = sum(1 for record in records if record["char_class"] == "cjk")
    payload = {
        "task": "Stage27 Song13 human-review artifact package",
        "source_metadata": str(source_metadata),
        "out_dir": str(out_dir),
        "overview_contact_sheet": str(overview_contact_sheet),
        "glyph_count": len(records),
        "cjk_glyph_count": cjk_count,
        "category_count": category_count,
        "image_order": ["source", "stage24_adapted", "stage24_predicted", "stage25_predicted", "stage26_predicted", "target"],
        "models": {
            "stage24": {"metadata": str(stage24_metadata), "candidate": stage24_name},
            "stage25": {"metadata": str(stage25_metadata), "candidate": stage25_name},
            "stage26": {"metadata": str(stage26_metadata), "candidate": stage26_name},
        },
        "categories": category_payload,
        "interpretation_notes": [
            "This stage is for human review and constraint diagnostics, not target-shape ranking.",
            "Stage24 can add outside-source ge2 pixels; Stage25 and Stage26 keep predicted ge2 equal to source.",
            "Use the contact sheets to judge readability, gray2 behavior, shadow density, and hole/counter fill risk.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Song13ReviewExport(
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        overview_contact_sheet=str(overview_contact_sheet),
        glyph_count=len(records),
        cjk_glyph_count=cjk_count,
        category_count=len(categories),
    )
