from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import BinaryMetrics, compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name, levels_to_mask, summarize_group
from font_machine_learn.paths import (
    CJK_EDGES_CONTACT,
    CJK_EDGES_DIR,
    CJK_EDGES_METADATA,
    CJK_EDGES_SEARCH,
    CJK_EDGES_WORST_CONTACT,
    SOURCE_METADATA,
)
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual
from font_machine_learn.weight_search import Mask, WeightRule, apply_weight_rule


@dataclass(frozen=True)
class CjkEdgeRule:
    name: str
    dx: int
    dy: int
    edge_offsets: str
    max_core_neighbors: int


@dataclass(frozen=True)
class CjkEdgeRuleSummary:
    name: str
    dx: int
    dy: int
    edge_offsets: str
    max_core_neighbors: int
    groups: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class CjkEdgesExport:
    source_metadata: str
    out_dir: str
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


def offset_set(name: str) -> tuple[tuple[int, int], ...]:
    if name == "none":
        return ()
    if name == "right":
        return ((1, 0),)
    if name == "down":
        return ((0, 1),)
    if name == "right_down":
        return ((1, 0), (0, 1))
    if name == "horizontal":
        return ((-1, 0), (1, 0))
    if name == "vertical":
        return ((0, -1), (0, 1))
    if name == "cardinal":
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    raise ValueError(f"unknown edge offset set: {name}")


def default_cjk_edge_rules() -> list[CjkEdgeRule]:
    rules: list[CjkEdgeRule] = []
    for offsets in ("none", "right", "down", "right_down", "horizontal", "vertical", "cardinal"):
        for max_neighbors in (1, 2, 3, 8):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    name = f"{offsets}_n{max_neighbors}_dx{dx:+d}_dy{dy:+d}"
                    rules.append(CjkEdgeRule(name, dx, dy, offsets, max_neighbors))
    return rules


def count_core_neighbors(core: Mask, x: int, y: int) -> int:
    height = len(core)
    width = len(core[0]) if height else 0
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sx = x + dx
            sy = y + dy
            if 0 <= sx < width and 0 <= sy < height and core[sy][sx]:
                count += 1
    return count


def edge_levels(source_mask: Mask, rule: CjkEdgeRule) -> list[list[int]]:
    core = apply_weight_rule(source_mask, WeightRule("core", rule.dx, rule.dy, "none"))
    height = len(core)
    width = len(core[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(height):
        for x in range(width):
            if core[y][x]:
                levels[y][x] = 3

    for y in range(height):
        for x in range(width):
            if not core[y][x]:
                continue
            for ox, oy in offset_set(rule.edge_offsets):
                sx = x + ox
                sy = y + oy
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                if levels[sy][sx] != 0:
                    continue
                if count_core_neighbors(core, sx, sy) <= rule.max_core_neighbors:
                    levels[sy][sx] = 2

    for y in range(height):
        for x in range(width):
            if not core[y][x]:
                continue
            sx = x + 1
            sy = y + 1
            if 0 <= sx < width and 0 <= sy < height and levels[sy][sx] == 0:
                levels[sy][sx] = 1
    return levels


def score_rule(glyphs: list[dict], rule: CjkEdgeRule, *, limit: int | None = None) -> CjkEdgeRuleSummary:
    selected = glyphs[:limit] if limit is not None else glyphs
    binary_by_group: dict[str, list[BinaryMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    for glyph in selected:
        char_class = classify_glyph(list(glyph["chars"]))
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        predicted_levels = edge_levels(source_mask, rule)
        binary = compare_masks(levels_to_mask(predicted_levels), target_mask)
        visual = compare_visual(predicted_levels, target_levels)
        for name in ("all", group_name(char_class)):
            binary_by_group[name].append(binary)
            visual_by_group[name].append(visual)
    return CjkEdgeRuleSummary(
        name=rule.name,
        dx=rule.dx,
        dy=rule.dy,
        edge_offsets=rule.edge_offsets,
        max_core_neighbors=rule.max_core_neighbors,
        groups={
            name: asdict(summarize_group(binary_by_group[name], visual_by_group[name]))
            for name in ("all", "cjk", "non_cjk")
        },
    )


def export_cjk_edges_baseline(
    source_metadata: Path = SOURCE_METADATA,
    *,
    out_dir: Path = CJK_EDGES_DIR,
    metadata_json: Path = CJK_EDGES_METADATA,
    search_json: Path = CJK_EDGES_SEARCH,
    contact_sheet: Path = CJK_EDGES_CONTACT,
    worst_contact_sheet: Path = CJK_EDGES_WORST_CONTACT,
    search_limit: int | None = None,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> CjkEdgesExport:
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
    worst_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    rules = default_cjk_edge_rules()
    summaries = [score_rule(glyphs, rule, limit=search_limit) for rule in rules]
    summaries.sort(
        key=lambda item: (
            float(item.groups["cjk"]["visual_score"]),
            float(item.groups["cjk"]["foreground_f1"]),
            float(item.groups["cjk"]["foreground_iou"]),
        ),
        reverse=True,
    )
    rule_by_name = {rule.name: rule for rule in rules}
    best_rule = rule_by_name[summaries[0].name]

    binary_by_group: dict[str, list[BinaryMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    records: list[BaselineGlyphMetrics] = []
    cjk_records: list[BaselineGlyphMetrics] = []
    glyph_payloads: list[dict] = []
    visual_by_index: dict[int, float] = {}
    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        predicted_levels = edge_levels(source_mask, best_rule)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        binary = compare_masks(levels_to_mask(predicted_levels), target_mask)
        visual = compare_visual(predicted_levels, target_levels)
        visual_by_index[index] = visual.visual_score
        for name in ("all", group_name(char_class)):
            binary_by_group[name].append(binary)
            visual_by_group[name].append(visual)
        record = BaselineGlyphMetrics(
            index=index,
            codes=list(glyph["codes"]),
            chars=chars,
            source_png=str(glyph["source_png"]),
            predicted_png=str(predicted_png),
            target_png=str(glyph["target_png"]),
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
        "source_metadata": str(source_metadata),
        "search_limit": search_limit,
        "selection": "sort by cjk visual_score, then cjk foreground_f1, then cjk foreground_iou",
        "rules": [asdict(summary) for summary in summaries],
    }
    metadata_payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(records),
        "cjk_glyph_count": len(cjk_records),
        "best_rule": asdict(summaries[0]),
        "style_constraints": {
            "core": "shifted source ink level 3",
            "edge_transition": "neighbor-limited source-adjacent edge/anti-alias pixels level 2",
            "shadow": "fixed right-down (1,1) level 1 from shifted source core",
        },
        "groups": groups,
        "out_dir": str(out_dir),
        "contact_sheet": str(contact_sheet),
        "worst_contact_sheet": str(worst_contact_sheet),
        "glyphs": glyph_payloads,
    }
    search_json.write_text(json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return CjkEdgesExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
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
