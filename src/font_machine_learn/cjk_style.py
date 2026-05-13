from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import (
    BaselineGlyphMetrics,
    image_to_target_levels,
    levels_to_image,
)
from font_machine_learn.binary_diagnostic import BinaryMetrics, compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.nftr import checkerboard
from font_machine_learn.paths import (
    CJK_STYLE_CONTACT,
    CJK_STYLE_DIR,
    CJK_STYLE_METADATA,
    CJK_STYLE_SEARCH,
    CJK_STYLE_WORST_CONTACT,
    SOURCE_METADATA,
)
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual
from font_machine_learn.weight_search import Mask, WeightRule, apply_weight_rule


@dataclass(frozen=True)
class CjkStyleRule:
    name: str
    dx: int
    dy: int
    edge_kernel: str


@dataclass(frozen=True)
class GroupSummary:
    glyph_count: int
    foreground_f1: float
    foreground_iou: float
    foreground_precision: float
    foreground_recall: float
    visual_score: float
    ink_f1: float
    shadow_f1: float
    visual_foreground_iou: float
    pixel_accuracy: float
    mean_absolute_error: float


@dataclass(frozen=True)
class CjkStyleRuleSummary:
    name: str
    dx: int
    dy: int
    edge_kernel: str
    groups: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class CjkStyleExport:
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


def default_cjk_style_rules() -> list[CjkStyleRule]:
    rules: list[CjkStyleRule] = []
    for kernel in ("none", "right_down", "horizontal", "vertical", "cardinal_light"):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rules.append(CjkStyleRule(f"{kernel}_dx{dx:+d}_dy{dy:+d}", dx, dy, kernel))
    return rules


def edge_transition_mask(mask: Mask, kernel: str) -> Mask:
    if kernel == "none":
        return [row[:] for row in mask]
    if kernel == "horizontal":
        offsets = ((0, 0), (1, 0), (-1, 0))
    elif kernel == "vertical":
        offsets = ((0, 0), (0, 1), (0, -1))
    elif kernel == "right_down":
        offsets = ((0, 0), (1, 0), (0, 1))
    elif kernel == "cardinal_light":
        offsets = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
    else:
        raise ValueError(f"unknown edge transition kernel: {kernel}")

    height = len(mask)
    width = len(mask[0]) if height else 0
    out = [[False for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            for dx, dy in offsets:
                tx = x + dx
                ty = y + dy
                if 0 <= tx < width and 0 <= ty < height:
                    out[ty][tx] = True
    return out


def style_levels(source_mask: Mask, rule: CjkStyleRule) -> list[list[int]]:
    shifted_core = apply_weight_rule(source_mask, WeightRule("core", rule.dx, rule.dy, "none"))
    edge = apply_weight_rule(
        edge_transition_mask(source_mask, rule.edge_kernel),
        WeightRule("edge", rule.dx, rule.dy, "none"),
    )
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(height):
        for x in range(width):
            if edge[y][x]:
                levels[y][x] = 2
            if shifted_core[y][x]:
                levels[y][x] = 3

    for y in range(height):
        for x in range(width):
            if not shifted_core[y][x]:
                continue
            sx = x + 1
            sy = y + 1
            if 0 <= sx < width and 0 <= sy < height and levels[sy][sx] == 0:
                levels[sy][sx] = 1
    return levels


def levels_to_mask(levels: list[list[int]]) -> Mask:
    return [[value > 0 for value in row] for row in levels]


def _mean_binary(records: list[BinaryMetrics], attr: str) -> float:
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def _mean_visual(records: list[VisualMetrics], attr: str) -> float:
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def summarize_group(binary: list[BinaryMetrics], visual: list[VisualMetrics]) -> GroupSummary:
    if not binary or not visual:
        return GroupSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return GroupSummary(
        glyph_count=len(binary),
        foreground_f1=_mean_binary(binary, "foreground_f1"),
        foreground_iou=_mean_binary(binary, "foreground_iou"),
        foreground_precision=_mean_binary(binary, "foreground_precision"),
        foreground_recall=_mean_binary(binary, "foreground_recall"),
        visual_score=_mean_visual(visual, "visual_score"),
        ink_f1=_mean_visual(visual, "ink_f1"),
        shadow_f1=_mean_visual(visual, "shadow_f1"),
        visual_foreground_iou=_mean_visual(visual, "foreground_iou"),
        pixel_accuracy=_mean_visual(visual, "pixel_accuracy"),
        mean_absolute_error=_mean_visual(visual, "mean_absolute_error"),
    )


def group_name(char_class: str) -> str:
    return "cjk" if char_class == "cjk" else "non_cjk"


def score_rule(glyphs: list[dict], rule: CjkStyleRule, *, limit: int | None = None) -> CjkStyleRuleSummary:
    selected = glyphs[:limit] if limit is not None else glyphs
    binary_by_group: dict[str, list[BinaryMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    for glyph in selected:
        char_class = classify_glyph(list(glyph["chars"]))
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        predicted_levels = style_levels(source_mask, rule)
        binary = compare_masks(levels_to_mask(predicted_levels), target_mask)
        visual = compare_visual(predicted_levels, target_levels)
        groups = ("all", group_name(char_class))
        for name in groups:
            binary_by_group[name].append(binary)
            visual_by_group[name].append(visual)
    return CjkStyleRuleSummary(
        name=rule.name,
        dx=rule.dx,
        dy=rule.dy,
        edge_kernel=rule.edge_kernel,
        groups={
            name: asdict(summarize_group(binary_by_group[name], visual_by_group[name]))
            for name in ("all", "cjk", "non_cjk")
        },
    )


def _make_contact_sheet(
    records: list[BaselineGlyphMetrics],
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
    group_h = tile_h * 3 + pad * 2
    rows = (len(records) + columns - 1) // columns
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, rows * group_h + (rows + 1) * pad),
        max(2, scale * 2),
    )

    for position, record in enumerate(records):
        x = pad + (position % columns) * (tile_w + pad)
        y = pad + (position // columns) * (group_h + pad)
        images = (
            Image.open(record.source_png).convert("RGBA"),
            Image.open(record.predicted_png).convert("RGBA"),
            Image.open(record.target_png).convert("RGBA"),
        )
        for row, image in enumerate(images):
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + row * (tile_h + pad)))
    sheet.save(out_path)


def export_cjk_style_baseline(
    source_metadata: Path = SOURCE_METADATA,
    *,
    out_dir: Path = CJK_STYLE_DIR,
    metadata_json: Path = CJK_STYLE_METADATA,
    search_json: Path = CJK_STYLE_SEARCH,
    contact_sheet: Path = CJK_STYLE_CONTACT,
    worst_contact_sheet: Path = CJK_STYLE_WORST_CONTACT,
    search_limit: int | None = None,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> CjkStyleExport:
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

    rules = default_cjk_style_rules()
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
    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        predicted_levels = style_levels(source_mask, best_rule)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        binary = compare_masks(levels_to_mask(predicted_levels), target_mask)
        visual = compare_visual(predicted_levels, target_levels)
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
    worst_cjk = sorted(
        cjk_records,
        key=lambda record: glyph_payloads[record.index]["visual"]["visual_score"],
    )[:worst_count]
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
            "edge_transition": "source-adjacent edge/anti-alias transition pixels level 2",
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

    return CjkStyleExport(
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
