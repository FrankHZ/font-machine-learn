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
    WQY_ALIGNMENT_CONTACT,
    WQY_ALIGNMENT_DIR,
    WQY_ALIGNMENT_METADATA,
    WQY_ALIGNMENT_SEARCH,
)
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.target_mask_compare import levels_to_threshold_mask, mask_has_foreground
from font_machine_learn.weight_search import Mask, WeightRule, apply_weight_rule, default_weight_rules


TARGET_MODES = ("visible", "ge2", "eq3")


@dataclass(frozen=True)
class AlignmentExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_visible_rule: str
    best_ge2_rule: str
    best_eq3_rule: str
    cjk_visible_f1: float
    cjk_ge2_f1: float
    cjk_eq3_f1: float


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


def group_names(char_class: str, source_nonempty: bool) -> list[str]:
    names = ["all"]
    names.append("cjk" if char_class == "cjk" else "non_cjk")
    if source_nonempty:
        names.append("source_nonempty")
        names.append("cjk_source_nonempty" if char_class == "cjk" else "non_cjk_source_nonempty")
    return names


def target_masks_for_glyph(glyph: dict) -> dict[str, Mask]:
    levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
    return {mode: levels_to_threshold_mask(levels, mode) for mode in TARGET_MODES}


def score_rule(glyphs: list[dict], rule: WeightRule, *, limit: int | None = None) -> dict:
    selected = glyphs[:limit] if limit is not None else glyphs
    grouped: dict[str, dict[str, list[BinaryMetrics]]] = {
        mode: {
            "all": [],
            "cjk": [],
            "non_cjk": [],
            "source_nonempty": [],
            "cjk_source_nonempty": [],
            "non_cjk_source_nonempty": [],
        }
        for mode in TARGET_MODES
    }
    for glyph in selected:
        char_class = classify_glyph(list(glyph["chars"]))
        source_mask = image_to_mask(glyph["source_png"])
        source_nonempty = mask_has_foreground(source_mask)
        adjusted = apply_weight_rule(source_mask, rule)
        targets = target_masks_for_glyph(glyph)
        groups = group_names(char_class, source_nonempty)
        for mode in TARGET_MODES:
            metrics = compare_masks(adjusted, targets[mode])
            for group in groups:
                grouped[mode][group].append(metrics)
    return {
        "name": rule.name,
        "dx": rule.dx,
        "dy": rule.dy,
        "kernel": rule.kernel,
        "modes": {
            mode: {group: summarize(records) for group, records in grouped[mode].items()}
            for mode in TARGET_MODES
        },
    }


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
    image_keys = (
        "source_png",
        "visible_aligned_png",
        "visible_target_png",
        "ge2_aligned_png",
        "ge2_target_png",
        "eq3_aligned_png",
        "eq3_target_png",
        "target_png",
    )
    group_h = tile_h * len(image_keys) + pad * (len(image_keys) - 1)
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (
            columns * tile_w + (columns + 1) * pad,
            sheet_rows * group_h + (sheet_rows + 1) * pad,
        ),
        max(2, scale * 2),
    )
    for position, row in enumerate(rows):
        x = pad + (position % columns) * (tile_w + pad)
        y = pad + (position // columns) * (group_h + pad)
        for image_index, key in enumerate(image_keys):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + image_index * (tile_h + pad)))
    sheet.save(out_path)


def export_wqy_alignment_diagnostic(
    source_metadata: Path = SOURCE_METADATA,
    *,
    out_dir: Path = WQY_ALIGNMENT_DIR,
    metadata_json: Path = WQY_ALIGNMENT_METADATA,
    search_json: Path = WQY_ALIGNMENT_SEARCH,
    contact_sheet: Path = WQY_ALIGNMENT_CONTACT,
    search_limit: int | None = None,
    contact_count: int = 96,
    scale: int = 4,
    columns: int = 12,
    pad: int = 1,
) -> AlignmentExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    search_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    rules = default_weight_rules()
    summaries = [score_rule(glyphs, rule, limit=search_limit) for rule in rules]
    best_by_mode: dict[str, dict] = {}
    rule_by_name = {rule.name: rule for rule in rules}
    for mode in TARGET_MODES:
        best_by_mode[mode] = max(
            summaries,
            key=lambda item: (
                float(item["modes"][mode]["cjk_source_nonempty"]["foreground_f1"]),
                float(item["modes"][mode]["cjk_source_nonempty"]["foreground_iou"]),
            ),
        )

    grouped: dict[str, dict[str, list[BinaryMetrics]]] = {
        mode: {
            "all": [],
            "cjk": [],
            "non_cjk": [],
            "source_nonempty": [],
            "cjk_source_nonempty": [],
            "non_cjk_source_nonempty": [],
        }
        for mode in TARGET_MODES
    }
    rows: list[dict] = []
    cjk_count = 0
    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        if char_class == "cjk":
            cjk_count += 1
        source_mask = image_to_mask(glyph["source_png"])
        source_nonempty = mask_has_foreground(source_mask)
        targets = target_masks_for_glyph(glyph)
        row = {
            "index": index,
            "codes": list(glyph["codes"]),
            "chars": chars,
            "char_class": char_class,
            "source_nonempty": source_nonempty,
            "source_png": str(glyph["source_png"]),
            "target_png": str(glyph["target_png"]),
            "modes": {},
        }
        groups = group_names(char_class, source_nonempty)
        for mode in TARGET_MODES:
            mode_dir = out_dir / mode
            target_dir = out_dir / f"target_{mode}"
            mode_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            rule = rule_by_name[str(best_by_mode[mode]["name"])]
            adjusted = apply_weight_rule(source_mask, rule)
            adjusted_png = mode_dir / f"glyph_{index:04d}.png"
            target_png = target_dir / f"glyph_{index:04d}.png"
            mask_to_image(adjusted).save(adjusted_png)
            mask_to_image(targets[mode]).save(target_png)
            metrics = compare_masks(adjusted, targets[mode])
            for group in groups:
                grouped[mode][group].append(metrics)
            row[f"{mode}_aligned_png"] = str(adjusted_png)
            row[f"{mode}_target_png"] = str(target_png)
            row["modes"][mode] = {
                "best_rule": str(best_by_mode[mode]["name"]),
                "aligned_png": str(adjusted_png),
                "target_mask_png": str(target_png),
                "metrics": asdict(metrics),
            }
        rows.append(row)

    selected = [row for row in rows if row["char_class"] == "cjk" and row["source_nonempty"]][:contact_count]
    _make_contact_sheet(
        selected,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    groups = {
        mode: {name: summarize(records) for name, records in grouped[mode].items()}
        for mode in TARGET_MODES
    }
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(rows),
        "cjk_glyph_count": cjk_count,
        "target_modes": {
            "visible": "target level > 0",
            "ge2": "target level >= 2",
            "eq3": "target level == 3",
        },
        "selection": "best rule per target mode by cjk_source_nonempty foreground_f1, then IoU",
        "best_by_mode": best_by_mode,
        "groups": groups,
        "out_dir": str(out_dir),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_order": [
            "wqy_source",
            "visible_aligned",
            "visible_target",
            "ge2_aligned",
            "ge2_target",
            "eq3_aligned",
            "eq3_target",
            "target_2bpp",
        ],
        "glyphs": rows,
    }
    search_json.write_text(
        json.dumps(
            {
                "source_metadata": str(source_metadata),
                "search_limit": search_limit,
                "rules": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return AlignmentExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        search_json=str(search_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(rows),
        cjk_glyph_count=cjk_count,
        best_visible_rule=str(best_by_mode["visible"]["name"]),
        best_ge2_rule=str(best_by_mode["ge2"]["name"]),
        best_eq3_rule=str(best_by_mode["eq3"]["name"]),
        cjk_visible_f1=float(groups["visible"]["cjk_source_nonempty"]["foreground_f1"]),
        cjk_ge2_f1=float(groups["ge2"]["cjk_source_nonempty"]["foreground_f1"]),
        cjk_eq3_f1=float(groups["eq3"]["cjk_source_nonempty"]["foreground_f1"]),
    )
