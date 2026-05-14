from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    BOUNDARY_RULE_CONTACT,
    BOUNDARY_RULE_DIR,
    BOUNDARY_RULE_METADATA,
    BOUNDARY_RULE_SEARCH,
    BOUNDARY_RULE_WORST_CONTACT,
    TARGET_METADATA,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import compare_visual


Mask = list[list[bool]]


@dataclass(frozen=True)
class BoundaryRule:
    name: str
    min_inside_neighbors: int
    edge_band: int
    diagonal_shadow: bool


@dataclass(frozen=True)
class BoundaryRuleExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    worst_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_rule: str
    cjk_visual_score: float
    cjk_ink_f1: float
    cjk_shadow_f1: float


def inside(mask: Mask, x: int, y: int) -> bool:
    return 0 <= y < len(mask) and 0 <= x < (len(mask[0]) if mask else 0) and mask[y][x]


def inside_neighbors(mask: Mask, x: int, y: int) -> int:
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            if inside(mask, x + dx, y + dy):
                count += 1
    return count


def distance_to_outside(mask: Mask, x: int, y: int) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    best = width + height
    for ty in range(height):
        for tx in range(width):
            if mask[ty][tx]:
                continue
            best = min(best, abs(tx - x) + abs(ty - y))
    return best


def level2_feature_stats(glyphs: list[dict]) -> dict:
    buckets = {
        "target2_by_inside_neighbors": {str(i): 0 for i in range(9)},
        "target3_by_inside_neighbors": {str(i): 0 for i in range(9)},
        "target2_by_distance_to_outside": {},
        "target3_by_distance_to_outside": {},
    }
    totals = {"level2": 0, "level3": 0}
    for glyph in glyphs:
        if classify_glyph(list(glyph["chars"])) != "cjk":
            continue
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        source_mask = levels_to_threshold_mask(target_levels, "ge2")
        for y, row in enumerate(target_levels):
            for x, value in enumerate(row):
                if value not in (2, 3):
                    continue
                neighbors = inside_neighbors(source_mask, x, y)
                distance = distance_to_outside(source_mask, x, y)
                key = "level2" if value == 2 else "level3"
                totals[key] += 1
                buckets[f"target{value}_by_inside_neighbors"][str(neighbors)] += 1
                distance_key = str(distance)
                buckets[f"target{value}_by_distance_to_outside"].setdefault(distance_key, 0)
                buckets[f"target{value}_by_distance_to_outside"][distance_key] += 1
    return {"totals": totals, **buckets}


def default_boundary_rules() -> list[BoundaryRule]:
    rules: list[BoundaryRule] = []
    for min_neighbors in range(1, 9):
        for edge_band in (0, 1, 2, 3):
            for diagonal_shadow in (False, True):
                name = f"n{min_neighbors}_band{edge_band}_{'diag' if diagonal_shadow else 'plain'}"
                rules.append(BoundaryRule(name, min_neighbors, edge_band, diagonal_shadow))
    return rules


def boundary_levels(target_levels: list[list[int]], rule: BoundaryRule) -> list[list[int]]:
    source_mask = levels_to_threshold_mask(target_levels, "ge2")
    height = len(target_levels)
    width = len(target_levels[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]

    for y in range(height):
        for x in range(width):
            if not source_mask[y][x]:
                continue
            neighbors = inside_neighbors(source_mask, x, y)
            distance = distance_to_outside(source_mask, x, y)
            if neighbors >= rule.min_inside_neighbors and distance > rule.edge_band:
                levels[y][x] = 3
            else:
                levels[y][x] = 2

    for y in range(height):
        for x in range(width):
            if levels[y][x] != 0:
                continue
            shadow = inside(source_mask, x - 1, y - 1)
            if rule.diagonal_shadow:
                shadow = shadow or inside(source_mask, x - 1, y) or inside(source_mask, x, y - 1)
            if shadow:
                levels[y][x] = 1
    return levels


def _mean(records: list, attr: str) -> float:
    if not records:
        return 0.0
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def _summarize(visual_by_group: dict[str, list]) -> dict[str, dict[str, float | int]]:
    return {
        group: {
            "glyph_count": len(records),
            "visual_score": _mean(records, "visual_score"),
            "ink_f1": _mean(records, "ink_f1"),
            "shadow_f1": _mean(records, "shadow_f1"),
            "foreground_iou": _mean(records, "foreground_iou"),
            "pixel_accuracy": _mean(records, "pixel_accuracy"),
            "mean_absolute_error": _mean(records, "mean_absolute_error"),
        }
        for group, records in visual_by_group.items()
    }


def score_rule(glyphs: list[dict], rule: BoundaryRule, *, limit: int | None = None) -> dict:
    selected = glyphs[:limit] if limit is not None else glyphs
    visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
    for glyph in selected:
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        predicted_levels = boundary_levels(target_levels, rule)
        visual = compare_visual(predicted_levels, target_levels)
        for group in ("all", group_name(char_class)):
            visual_by_group[group].append(visual)
    return {
        **asdict(rule),
        "groups": _summarize(visual_by_group),
    }


def export_boundary_rules(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = BOUNDARY_RULE_DIR,
    metadata_json: Path = BOUNDARY_RULE_METADATA,
    search_json: Path = BOUNDARY_RULE_SEARCH,
    contact_sheet: Path = BOUNDARY_RULE_CONTACT,
    worst_contact_sheet: Path = BOUNDARY_RULE_WORST_CONTACT,
    search_limit: int | None = None,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> BoundaryRuleExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    search_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    worst_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    rules = default_boundary_rules()
    summaries = [score_rule(glyphs, rule, limit=search_limit) for rule in rules]
    summaries.sort(
        key=lambda item: (
            float(item["groups"]["cjk"]["visual_score"]),
            float(item["groups"]["cjk"]["ink_f1"]),
            float(item["groups"]["cjk"]["shadow_f1"]),
        ),
        reverse=True,
    )
    rule_by_name = {rule.name: rule for rule in rules}
    best_rule = rule_by_name[str(summaries[0]["name"])]

    visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
    records: list[BaselineGlyphMetrics] = []
    cjk_records: list[BaselineGlyphMetrics] = []
    glyph_payloads: list[dict] = []
    visual_by_index: dict[int, float] = {}
    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        predicted_levels = boundary_levels(target_levels, best_rule)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        visual = compare_visual(predicted_levels, target_levels)
        visual_by_index[index] = visual.visual_score
        for group in ("all", group_name(char_class)):
            visual_by_group[group].append(visual)
        source_png = out_dir.parent / "source_ge2" / f"glyph_{index:04d}.png"
        source_png.parent.mkdir(parents=True, exist_ok=True)
        levels_to_image([[3 if value >= 2 else 0 for value in row] for row in target_levels]).save(source_png)
        record = BaselineGlyphMetrics(
            index=index,
            codes=list(glyph["codes"]),
            chars=chars,
            source_png=str(source_png),
            predicted_png=str(predicted_png),
            target_png=str(glyph["png"]),
            pixel_accuracy=visual.pixel_accuracy,
            mean_absolute_error=visual.mean_absolute_error,
            foreground_iou=visual.foreground_iou,
        )
        records.append(record)
        if char_class == "cjk":
            cjk_records.append(record)
        glyph_payloads.append({**asdict(record), "char_class": char_class, "visual": visual.to_dict()})

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

    groups = _summarize(visual_by_group)
    search_payload = {
        "target_metadata": str(target_metadata),
        "search_limit": search_limit,
        "selection": "sort by cjk visual_score, then ink_f1, then shadow_f1",
        "feature_stats": level2_feature_stats(glyphs),
        "rules": summaries,
    }
    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(records),
        "cjk_glyph_count": len(cjk_records),
        "task": "explain ge2 source pixels as level 2 vs level 3 boundary",
        "best_rule": summaries[0],
        "groups": groups,
        "out_dir": str(out_dir),
        "contact_sheet": str(contact_sheet),
        "worst_contact_sheet": str(worst_contact_sheet),
        "glyphs": glyph_payloads,
    }
    search_json.write_text(json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return BoundaryRuleExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        search_json=str(search_json),
        contact_sheet=str(contact_sheet),
        worst_contact_sheet=str(worst_contact_sheet),
        glyph_count=len(records),
        cjk_glyph_count=len(cjk_records),
        best_rule=best_rule.name,
        cjk_visual_score=float(groups["cjk"]["visual_score"]),
        cjk_ink_f1=float(groups["cjk"]["ink_f1"]),
        cjk_shadow_f1=float(groups["cjk"]["shadow_f1"]),
    )
