from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import group_name
from font_machine_learn.external_eval import summarize_binary, summarize_visual
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    SONG13_SOURCE_LOCKED_CONTACT,
    SONG13_SOURCE_LOCKED_ERROR_CONTACT,
    SONG13_SOURCE_LOCKED_METADATA,
    SONG13_SOURCE_LOCKED_SEARCH,
    STAGE25_SONG13_SOURCE_LOCKED,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA, Mask, ge2_mask_to_image, make_adapter_contact_sheet
from font_machine_learn.song13_calibrated import mask_foreground_ratio, mean
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class SourceLockedRule:
    name: str
    edge_max_neighbors: int
    shadow_offsets: str
    shadow_from: str


@dataclass(frozen=True)
class Song13SourceLockedExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_rule: str
    best_cjk_visual_score: float
    best_source_deleted_ratio: float
    best_source_ge2_ratio: float


def default_rules() -> list[SourceLockedRule]:
    rules: list[SourceLockedRule] = []
    for edge_max in (-1, 1, 2, 3, 4):
        for shadow_offsets in ("diag", "right_down", "diag_plus_right", "diag_plus_down"):
            for shadow_from in ("source", "core"):
                edge_name = "all3" if edge_max < 0 else f"edge_n{edge_max}"
                name = f"{edge_name}_{shadow_offsets}_from_{shadow_from}"
                rules.append(SourceLockedRule(name, edge_max, shadow_offsets, shadow_from))
    return rules


def neighbor_count(mask: Mask, x: int, y: int) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sx = x + dx
            sy = y + dy
            if 0 <= sx < width and 0 <= sy < height and mask[sy][sx]:
                count += 1
    return count


def shadow_offset_set(name: str) -> tuple[tuple[int, int], ...]:
    if name == "diag":
        return ((1, 1),)
    if name == "right_down":
        return ((1, 0), (0, 1), (1, 1))
    if name == "diag_plus_right":
        return ((1, 1), (1, 0))
    if name == "diag_plus_down":
        return ((1, 1), (0, 1))
    raise ValueError(f"unknown shadow offset set: {name}")


def source_locked_levels(source_mask: Mask, rule: SourceLockedRule) -> list[list[int]]:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(height):
        for x in range(width):
            if not source_mask[y][x]:
                continue
            if rule.edge_max_neighbors >= 0 and neighbor_count(source_mask, x, y) <= rule.edge_max_neighbors:
                levels[y][x] = 2
            else:
                levels[y][x] = 3

    for y in range(height):
        for x in range(width):
            if not source_mask[y][x]:
                continue
            if rule.shadow_from == "core" and levels[y][x] != 3:
                continue
            for dx, dy in shadow_offset_set(rule.shadow_offsets):
                sx = x + dx
                sy = y + dy
                if 0 <= sx < width and 0 <= sy < height and levels[sy][sx] == 0:
                    levels[sy][sx] = 1
    return levels


def levels_to_mask(levels: list[list[int]]) -> Mask:
    return [[value > 0 for value in row] for row in levels]


def source_deleted_ratio(source_mask: Mask, predicted_levels: list[list[int]]) -> float:
    source_count = sum(1 for row in source_mask for value in row if value)
    if source_count == 0:
        return 0.0
    deleted = sum(
        1
        for y, row in enumerate(source_mask)
        for x, value in enumerate(row)
        if value and predicted_levels[y][x] == 0
    )
    return deleted / source_count


def source_level_ratios(source_mask: Mask, predicted_levels: list[list[int]]) -> dict[str, float]:
    source_count = sum(1 for row in source_mask for value in row if value)
    if source_count == 0:
        return {"level2": 0.0, "level3": 0.0}
    level2 = sum(
        1
        for y, row in enumerate(source_mask)
        for x, value in enumerate(row)
        if value and predicted_levels[y][x] == 2
    )
    level3 = sum(
        1
        for y, row in enumerate(source_mask)
        for x, value in enumerate(row)
        if value and predicted_levels[y][x] == 3
    )
    return {"level2": level2 / source_count, "level3": level3 / source_count}


def rule_quality(summary: dict) -> float:
    cjk = summary["groups"]["cjk"]
    contract = summary["source_contract"]["cjk"]
    # Keep this mostly visual, but make source deletion disqualifying if future rules add removal.
    return (
        0.48 * float(cjk["visual_score"])
        + 0.22 * float(cjk["ink_f1"])
        + 0.18 * float(cjk["shadow_f1"])
        + 0.12 * float(cjk["foreground_iou"])
        - 0.75 * float(contract["source_deleted_ratio"])
    )


def evaluate_rule(
    rule: SourceLockedRule,
    source_glyphs: list[dict],
    *,
    out_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    binary_by_group = {"all": [], "cjk": [], "non_cjk": []}
    source_deleted_by_group = {"all": [], "cjk": [], "non_cjk": []}
    source_level2_by_group = {"all": [], "cjk": [], "non_cjk": []}
    source_level3_by_group = {"all": [], "cjk": [], "non_cjk": []}
    ratios = {
        "all": {"source": [], "predicted": [], "target": []},
        "cjk": {"source": [], "predicted": [], "target": []},
        "non_cjk": {"source": [], "predicted": [], "target": []},
    }
    glyph_payloads: list[dict] = []

    pred_dir = out_dir / rule.name / "predicted_2bpp" if out_dir else None
    source_ge2_dir = out_dir / rule.name / "source_ge2" if out_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    if source_ge2_dir:
        source_ge2_dir.mkdir(parents=True, exist_ok=True)

    for glyph in source_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = group_name(char_class)
        source_mask = image_to_mask(glyph["source_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_mask = levels_to_threshold_mask(target_levels, "visible")
        predicted_levels = source_locked_levels(source_mask, rule)
        predicted_mask = levels_to_mask(predicted_levels)

        predicted_png = None
        source_ge2_png = None
        if pred_dir is not None and source_ge2_dir is not None:
            predicted_png = pred_dir / f"glyph_{index:04d}.png"
            source_ge2_png = source_ge2_dir / f"glyph_{index:04d}.png"
            levels_to_image(predicted_levels).save(predicted_png)
            ge2_mask_to_image(source_mask).save(source_ge2_png)

        binary = compare_masks(predicted_mask, target_mask)
        visual = compare_visual(predicted_levels, target_levels)
        deleted = source_deleted_ratio(source_mask, predicted_levels)
        level_ratios = source_level_ratios(source_mask, predicted_levels)
        for key in ("all", group):
            visual_by_group[key].append(visual)
            binary_by_group[key].append(binary)
            source_deleted_by_group[key].append(deleted)
            source_level2_by_group[key].append(level_ratios["level2"])
            source_level3_by_group[key].append(level_ratios["level3"])
            ratios[key]["source"].append(mask_foreground_ratio(source_mask))
            ratios[key]["predicted"].append(mask_foreground_ratio(predicted_mask))
            ratios[key]["target"].append(mask_foreground_ratio(target_mask))

        glyph_payload = {
            "index": index,
            "codes": list(glyph["codes"]),
            "chars": chars,
            "original_source_png": str(glyph["source_png"]),
            "source_png": str(source_ge2_png) if source_ge2_png else str(glyph["source_png"]),
            "predicted_png": str(predicted_png) if predicted_png else "",
            "target_png": str(glyph["target_png"]),
            "char_class": char_class,
            "source_deleted_ratio": deleted,
            "source_level_ratio": level_ratios,
            "binary": asdict(binary),
            "visual": visual.to_dict(),
        }
        glyph_payloads.append(glyph_payload)

    groups = {key: summarize_visual(visual_by_group[key]) for key in ("all", "cjk", "non_cjk")}
    summary = {
        "rule": asdict(rule),
        "groups": groups,
        "binary": {key: summarize_binary(binary_by_group[key]) for key in ("all", "cjk", "non_cjk")},
        "foreground_ratio": {
            key: {
                "source": mean(ratios[key]["source"]),
                "predicted": mean(ratios[key]["predicted"]),
                "target_visible": mean(ratios[key]["target"]),
            }
            for key in ("all", "cjk", "non_cjk")
        },
        "source_contract": {
            key: {
                "source_deleted_ratio": mean(source_deleted_by_group[key]),
                "source_level2_ratio": mean(source_level2_by_group[key]),
                "source_level3_ratio": mean(source_level3_by_group[key]),
            }
            for key in ("all", "cjk", "non_cjk")
        },
    }
    summary["cjk_quality_score"] = rule_quality(summary)
    return summary, glyph_payloads


def export_song13_source_locked(
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE25_SONG13_SOURCE_LOCKED,
    metadata_json: Path = SONG13_SOURCE_LOCKED_METADATA,
    search_json: Path = SONG13_SOURCE_LOCKED_SEARCH,
    contact_sheet: Path = SONG13_SOURCE_LOCKED_CONTACT,
    error_contact_sheet: Path = SONG13_SOURCE_LOCKED_ERROR_CONTACT,
    search_limit: int | None = 512,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13SourceLockedExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    if not source_metadata.exists():
        raise FileNotFoundError(
            f"Source metadata not found: {source_metadata}. "
            "Render the Song13 baseline with scripts/render_source_glyphs.py first."
        )

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    source_glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    search_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    search_glyphs = source_glyphs[:search_limit] if search_limit is not None else source_glyphs
    rules = default_rules()
    search_summaries = [evaluate_rule(rule, search_glyphs)[0] for rule in rules]
    search_summaries.sort(key=lambda item: float(item["cjk_quality_score"]), reverse=True)
    rule_by_name = {rule.name: rule for rule in rules}
    best_rule = rule_by_name[search_summaries[0]["rule"]["name"]]

    best_summary, glyphs = evaluate_rule(best_rule, source_glyphs, out_dir=out_dir)
    cjk_records = [glyph for glyph in glyphs if glyph["char_class"] == "cjk"]
    make_adapter_contact_sheet(
        cjk_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst_records = sorted(cjk_records, key=lambda record: float(record["visual"]["visual_score"]))[:worst_count]
    make_adapter_contact_sheet(
        worst_records,
        out_path=error_contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    search_payload = {
        "source_metadata": str(source_metadata),
        "target_metadata": str(target_metadata),
        "search_limit": search_limit,
        "selection": "sort by cjk_quality_score; all rules preserve Song13 source pixels",
        "rules": search_summaries,
    }
    metadata_payload = {
        "source_metadata": str(source_metadata),
        "target_metadata": str(target_metadata),
        "glyph_count": len(source_glyphs),
        "cjk_glyph_count": len(cjk_records),
        "task": "source-locked Song13 1bpp mask -> NFTR-style 2bpp layer assignment",
        "source_contract": "Song13 source pixels are never deleted; source pixels become level 2 or 3, and shadows are added outside source.",
        "best_rule": best_summary["rule"],
        "best_summary": best_summary,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_order": ["original_source", "source_ge2", "predicted_2bpp", "target_2bpp"],
        "glyphs": glyphs,
        "interpretation_notes": [
            "This stage intentionally stops using target ge2 as a shape adapter target.",
            "Scores against target glyphs are supporting diagnostics only because Song13 and target glyph shapes differ.",
            "Use the contact sheet to judge whether the style assignment is readable while preserving source strokes.",
        ],
    }
    search_json.write_text(json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return Song13SourceLockedExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        search_json=str(search_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(source_glyphs),
        cjk_glyph_count=len(cjk_records),
        best_rule=str(best_summary["rule"]["name"]),
        best_cjk_visual_score=float(best_summary["groups"]["cjk"]["visual_score"]),
        best_source_deleted_ratio=float(best_summary["source_contract"]["cjk"]["source_deleted_ratio"]),
        best_source_ge2_ratio=float(best_summary["foreground_ratio"]["cjk"]["source"]),
    )
