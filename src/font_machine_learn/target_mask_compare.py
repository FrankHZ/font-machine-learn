from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels
from font_machine_learn.binary_diagnostic import BinaryMetrics, compare_masks, image_to_mask, mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.nftr import checkerboard
from font_machine_learn.paths import (
    SOURCE_METADATA,
    TARGET_EQ3_1BPP_DIR,
    TARGET_GE2_1BPP_DIR,
    TARGET_MASK_COMPARE_CONTACT,
    TARGET_MASK_COMPARE_METADATA,
)
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.weight_search import Mask


@dataclass(frozen=True)
class MaskCompareExport:
    source_metadata: str
    ge2_dir: str
    eq3_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    source_nonempty_count: int
    cjk_source_nonempty_count: int
    best_all: str
    best_cjk: str
    best_cjk_source_nonempty: str


def levels_to_threshold_mask(levels: list[list[int]], mode: str) -> Mask:
    if mode == "visible":
        return [[value > 0 for value in row] for row in levels]
    if mode == "ge2":
        return [[value >= 2 for value in row] for row in levels]
    if mode == "eq3":
        return [[value == 3 for value in row] for row in levels]
    raise ValueError(f"unknown target mask mode: {mode}")


def mask_has_foreground(mask: Mask) -> bool:
    return any(any(row) for row in mask)


def _mean(records: list[BinaryMetrics], attr: str) -> float:
    if not records:
        return 0.0
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def summarize(records: list[BinaryMetrics]) -> dict[str, float | int]:
    return {
        "glyph_count": len(records),
        "pixel_accuracy": _mean(records, "pixel_accuracy"),
        "foreground_precision": _mean(records, "foreground_precision"),
        "foreground_recall": _mean(records, "foreground_recall"),
        "foreground_f1": _mean(records, "foreground_f1"),
        "foreground_iou": _mean(records, "foreground_iou"),
        "false_positive_rate": _mean(records, "false_positive_rate"),
        "false_negative_rate": _mean(records, "false_negative_rate"),
    }


def group_name(char_class: str, source_nonempty: bool) -> list[str]:
    names = ["all"]
    names.append("cjk" if char_class == "cjk" else "non_cjk")
    if source_nonempty:
        names.append("source_nonempty")
        names.append("cjk_source_nonempty" if char_class == "cjk" else "non_cjk_source_nonempty")
    return names


def _make_contact_sheet(
    rows: list[dict],
    *,
    out_path: Path,
    scale: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    pad: int,
) -> None:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    group_h = tile_h * 4 + pad * 3
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (
            columns * tile_w + (columns + 1) * pad,
            sheet_rows * group_h + (sheet_rows + 1) * pad,
        ),
        max(2, scale * 2),
    )
    image_keys = ("source_png", "target_ge2_png", "target_eq3_png", "target_png")
    for position, row in enumerate(rows):
        x = pad + (position % columns) * (tile_w + pad)
        y = pad + (position // columns) * (group_h + pad)
        for image_index, key in enumerate(image_keys):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + image_index * (tile_h + pad)))
    sheet.save(out_path)


def export_target_mask_compare(
    source_metadata: Path = SOURCE_METADATA,
    *,
    ge2_dir: Path = TARGET_GE2_1BPP_DIR,
    eq3_dir: Path = TARGET_EQ3_1BPP_DIR,
    metadata_json: Path = TARGET_MASK_COMPARE_METADATA,
    contact_sheet: Path = TARGET_MASK_COMPARE_CONTACT,
    contact_count: int = 192,
    scale: int = 4,
    columns: int = 24,
    pad: int = 1,
) -> MaskCompareExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])

    ge2_dir.mkdir(parents=True, exist_ok=True)
    eq3_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, list[BinaryMetrics]]] = {
        mode: {
            "all": [],
            "cjk": [],
            "non_cjk": [],
            "source_nonempty": [],
            "cjk_source_nonempty": [],
            "non_cjk_source_nonempty": [],
        }
        for mode in ("ge2", "eq3")
    }
    rows: list[dict] = []
    cjk_source_nonempty_count = 0
    source_nonempty_count = 0

    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        source_mask = image_to_mask(glyph["source_png"])
        source_nonempty = mask_has_foreground(source_mask)
        if source_nonempty:
            source_nonempty_count += 1
            if char_class == "cjk":
                cjk_source_nonempty_count += 1

        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        ge2_mask = levels_to_threshold_mask(target_levels, "ge2")
        eq3_mask = levels_to_threshold_mask(target_levels, "eq3")
        ge2_png = ge2_dir / f"glyph_{index:04d}.png"
        eq3_png = eq3_dir / f"glyph_{index:04d}.png"
        mask_to_image(ge2_mask).save(ge2_png)
        mask_to_image(eq3_mask).save(eq3_png)

        metrics = {
            "ge2": compare_masks(source_mask, ge2_mask),
            "eq3": compare_masks(source_mask, eq3_mask),
        }
        groups = group_name(char_class, source_nonempty)
        for mode in ("ge2", "eq3"):
            for name in groups:
                grouped[mode][name].append(metrics[mode])

        rows.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": chars,
                "char_class": char_class,
                "source_nonempty": source_nonempty,
                "source_png": str(glyph["source_png"]),
                "target_png": str(glyph["target_png"]),
                "target_ge2_png": str(ge2_png),
                "target_eq3_png": str(eq3_png),
                "ge2": asdict(metrics["ge2"]),
                "eq3": asdict(metrics["eq3"]),
            }
        )

    selected = [
        row for row in rows if row["char_class"] == "cjk" and row["source_nonempty"]
    ][:contact_count]
    _make_contact_sheet(
        selected,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    summaries = {
        mode: {name: summarize(records) for name, records in grouped[mode].items()}
        for mode in ("ge2", "eq3")
    }
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(rows),
        "cjk_glyph_count": sum(1 for row in rows if row["char_class"] == "cjk"),
        "source_nonempty_count": source_nonempty_count,
        "cjk_source_nonempty_count": cjk_source_nonempty_count,
        "mask_modes": {
            "ge2": "target level >= 2",
            "eq3": "target level == 3",
        },
        "contact_sheet_order": ["wqy_source", "target_ge2", "target_eq3", "target_2bpp"],
        "ge2_dir": str(ge2_dir),
        "eq3_dir": str(eq3_dir),
        "contact_sheet": str(contact_sheet),
        "summaries": summaries,
        "glyphs": rows,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def best(group: str) -> str:
        ge2 = float(summaries["ge2"][group]["foreground_f1"])
        eq3 = float(summaries["eq3"][group]["foreground_f1"])
        return "ge2" if ge2 >= eq3 else "eq3"

    return MaskCompareExport(
        source_metadata=str(source_metadata),
        ge2_dir=str(ge2_dir),
        eq3_dir=str(eq3_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(rows),
        cjk_glyph_count=int(payload["cjk_glyph_count"]),
        source_nonempty_count=source_nonempty_count,
        cjk_source_nonempty_count=cjk_source_nonempty_count,
        best_all=best("all"),
        best_cjk=best("cjk"),
        best_cjk_source_nonempty=best("cjk_source_nonempty"),
    )
