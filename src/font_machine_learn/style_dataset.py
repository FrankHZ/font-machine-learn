from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask, mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_edges import count_core_neighbors
from font_machine_learn.cjk_style import _make_contact_sheet, group_name, levels_to_mask, summarize_group
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    STYLE_BASELINE_2BPP_DIR,
    STYLE_BASELINE_CONTACT,
    STYLE_BASELINE_SEARCH,
    STYLE_BASELINE_WORST_CONTACT,
    STYLE_INPUT_1BPP_DIR,
    STYLE_PAIRS_METADATA,
    TARGET_METADATA,
)
from font_machine_learn.visual_metrics import compare_visual
from font_machine_learn.weight_search import Mask, WeightRule, apply_weight_rule


@dataclass(frozen=True)
class StyleDatasetExport:
    target_metadata: str
    input_dir: str
    baseline_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    worst_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_rule: str
    cjk_foreground_f1: float
    cjk_foreground_iou: float
    cjk_visual_score: float
    all_visual_score: float
    non_cjk_visual_score: float


@dataclass(frozen=True)
class StyleLayerRule:
    name: str
    dx: int
    dy: int
    min_core_neighbors: int
    shadow_mode: str


def glyph_png(glyph: dict) -> str:
    return str(glyph.get("target_png") or glyph["png"])


def default_style_layer_rules() -> list[StyleLayerRule]:
    rules: list[StyleLayerRule] = []
    for min_neighbors in (0, 1, 2, 3, 4, 5):
        for shadow_mode in ("right_down", "none"):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    name = f"core_n{min_neighbors}_{shadow_mode}_dx{dx:+d}_dy{dy:+d}"
                    rules.append(StyleLayerRule(name, dx, dy, min_neighbors, shadow_mode))
    return rules


def core_from_visible_mask(mask: Mask, rule: StyleLayerRule) -> Mask:
    shifted = apply_weight_rule(mask, WeightRule("visible", rule.dx, rule.dy, "none"))
    if rule.min_core_neighbors <= 0:
        return shifted
    height = len(shifted)
    width = len(shifted[0]) if height else 0
    core = [[False for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            if shifted[y][x] and count_core_neighbors(shifted, x, y) >= rule.min_core_neighbors:
                core[y][x] = True
    return core


def style_levels_from_visible_mask(mask: Mask, rule: StyleLayerRule) -> list[list[int]]:
    core = core_from_visible_mask(mask, rule)
    height = len(mask)
    width = len(mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(height):
        for x in range(width):
            if mask[y][x]:
                levels[y][x] = 2
            if core[y][x] and mask[y][x]:
                levels[y][x] = 3

    if rule.shadow_mode == "right_down":
        for y in range(height):
            for x in range(width):
                if not mask[y][x] or core[y][x]:
                    continue
                ux = x - 1
                uy = y - 1
                if 0 <= ux < width and 0 <= uy < height and core[uy][ux] and mask[uy][ux]:
                    levels[y][x] = 1
    elif rule.shadow_mode != "none":
        raise ValueError(f"unknown shadow mode: {rule.shadow_mode}")

    return levels


def _score_style_rule(glyphs: list[dict], rule: StyleLayerRule, *, limit: int | None = None) -> dict:
    selected = glyphs[:limit] if limit is not None else glyphs
    binary_by_group = {"all": [], "cjk": [], "non_cjk": []}
    visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
    for glyph in selected:
        char_class = classify_glyph(list(glyph["chars"]))
        label_png = glyph_png(glyph)
        source_mask = image_to_mask(label_png)
        target_mask = source_mask
        target_levels = image_to_target_levels(Image.open(label_png).convert("RGBA"))
        predicted_levels = style_levels_from_visible_mask(source_mask, rule)
        binary = compare_masks(levels_to_mask(predicted_levels), target_mask)
        visual = compare_visual(predicted_levels, target_levels)
        for name in ("all", group_name(char_class)):
            binary_by_group[name].append(binary)
            visual_by_group[name].append(visual)
    return {
        "name": rule.name,
        "dx": rule.dx,
        "dy": rule.dy,
        "min_core_neighbors": rule.min_core_neighbors,
        "shadow_mode": rule.shadow_mode,
        "groups": {
            name: asdict(summarize_group(binary_by_group[name], visual_by_group[name]))
            for name in ("all", "cjk", "non_cjk")
        },
    }


def export_1bpp_style_dataset(
    target_metadata: Path = TARGET_METADATA,
    *,
    input_dir: Path = STYLE_INPUT_1BPP_DIR,
    baseline_dir: Path = STYLE_BASELINE_2BPP_DIR,
    metadata_json: Path = STYLE_PAIRS_METADATA,
    search_json: Path = STYLE_BASELINE_SEARCH,
    contact_sheet: Path = STYLE_BASELINE_CONTACT,
    worst_contact_sheet: Path = STYLE_BASELINE_WORST_CONTACT,
    search_limit: int | None = None,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> StyleDatasetExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    input_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    search_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    worst_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    rules = default_style_layer_rules()
    summaries = [_score_style_rule(glyphs, rule, limit=search_limit) for rule in rules]
    summaries.sort(
        key=lambda item: (
            float(item["groups"]["cjk"]["visual_score"]),
            float(item["groups"]["cjk"]["foreground_f1"]),
            float(item["groups"]["cjk"]["foreground_iou"]),
        ),
        reverse=True,
    )
    rule_by_name = {rule.name: rule for rule in rules}
    best_rule = rule_by_name[str(summaries[0]["name"])]

    binary_by_group = {"all": [], "cjk": [], "non_cjk": []}
    visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
    records: list[BaselineGlyphMetrics] = []
    cjk_records: list[BaselineGlyphMetrics] = []
    glyph_payloads: list[dict] = []
    visual_by_index: dict[int, float] = {}

    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        label_png = glyph_png(glyph)
        source_mask = image_to_mask(label_png)
        input_png = input_dir / f"glyph_{index:04d}.png"
        mask_to_image(source_mask).save(input_png)

        target_levels = image_to_target_levels(Image.open(label_png).convert("RGBA"))
        predicted_levels = style_levels_from_visible_mask(source_mask, best_rule)
        predicted_png = baseline_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        binary = compare_masks(levels_to_mask(predicted_levels), source_mask)
        visual = compare_visual(predicted_levels, target_levels)
        visual_by_index[index] = visual.visual_score
        for name in ("all", group_name(char_class)):
            binary_by_group[name].append(binary)
            visual_by_group[name].append(visual)

        record = BaselineGlyphMetrics(
            index=index,
            codes=list(glyph["codes"]),
            chars=chars,
            source_png=str(input_png),
            predicted_png=str(predicted_png),
            target_png=label_png,
            pixel_accuracy=visual.pixel_accuracy,
            mean_absolute_error=visual.mean_absolute_error,
            foreground_iou=visual.foreground_iou,
        )
        records.append(record)
        if char_class == "cjk":
            cjk_records.append(record)
        glyph_payloads.append(
            {
                **asdict(record),
                "char_class": char_class,
                "input_1bpp_png": str(input_png),
                "label_2bpp_png": label_png,
                "binary": asdict(binary),
                "visual": visual.to_dict(),
            }
        )

    _make_contact_sheet(
        cjk_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst_cjk = sorted(cjk_records, key=lambda record: visual_by_index[record.index])[:worst_count]
    _make_contact_sheet(
        worst_cjk,
        out_path=worst_contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    groups = {
        name: asdict(summarize_group(binary_by_group[name], visual_by_group[name]))
        for name in ("all", "cjk", "non_cjk")
    }
    search_payload = {
        "target_metadata": str(target_metadata),
        "search_limit": search_limit,
        "selection": "sort by cjk visual_score, then cjk foreground_f1, then cjk foreground_iou",
        "rules": summaries,
    }
    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(records),
        "cjk_glyph_count": len(cjk_records),
        "task": "1bpp glyph mask -> NFTR-style 2bpp layered glyph",
        "best_rule": summaries[0],
        "style_constraints": {
            "input": "target-derived 1bpp mask",
            "label": "original target 2bpp glyph",
            "visible_mask": "target-derived 1bpp mask contains all visible target pixels",
            "core": "searched dense subset of the 1bpp visible mask",
            "edge_transition": "visible mask pixels outside core default to level 2",
            "shadow": "visible non-core pixels right-down from core may become level 1",
        },
        "groups": groups,
        "input_dir": str(input_dir),
        "baseline_dir": str(baseline_dir),
        "contact_sheet": str(contact_sheet),
        "worst_contact_sheet": str(worst_contact_sheet),
        "glyphs": glyph_payloads,
    }
    search_json.write_text(json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return StyleDatasetExport(
        target_metadata=str(target_metadata),
        input_dir=str(input_dir),
        baseline_dir=str(baseline_dir),
        metadata_json=str(metadata_json),
        search_json=str(search_json),
        contact_sheet=str(contact_sheet),
        worst_contact_sheet=str(worst_contact_sheet),
        glyph_count=len(records),
        cjk_glyph_count=len(cjk_records),
        best_rule=best_rule.name,
        cjk_foreground_f1=float(groups["cjk"]["foreground_f1"]),
        cjk_foreground_iou=float(groups["cjk"]["foreground_iou"]),
        cjk_visual_score=float(groups["cjk"]["visual_score"]),
        all_visual_score=float(groups["all"]["visual_score"]),
        non_cjk_visual_score=float(groups["non_cjk"]["visual_score"]),
    )
