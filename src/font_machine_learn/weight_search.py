from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import (
    BaselineGlyphMetrics,
    image_to_target_levels,
    levels_to_image,
    make_baseline_contact_sheet,
    source_image_to_levels,
)
from font_machine_learn.binary_diagnostic import BinaryMetrics, compare_masks, image_to_mask, mask_to_image
from font_machine_learn.shadow_search import ShadowRule, apply_shadow_rule
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.visual_metrics import compare_visual


Mask = list[list[bool]]


@dataclass(frozen=True)
class WeightRule:
    name: str
    dx: int
    dy: int
    kernel: str


@dataclass(frozen=True)
class WeightRuleSummary:
    name: str
    dx: int
    dy: int
    kernel: str
    glyph_count: int
    foreground_f1: float
    foreground_iou: float
    foreground_precision: float
    foreground_recall: float
    pixel_accuracy: float


@dataclass(frozen=True)
class WeightSearchExport:
    source_metadata: str
    source_out_dir: str
    shadow_out_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    glyph_count: int
    best_rule: str
    source_foreground_f1: float
    source_foreground_iou: float
    shadow_visual_score: float
    shadow_foreground_iou: float
    shadow_pixel_accuracy: float
    shadow_mean_absolute_error: float


def default_weight_rules() -> list[WeightRule]:
    rules: list[WeightRule] = []
    for kernel in ("none", "right_down", "cardinal", "box"):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rules.append(WeightRule(f"{kernel}_dx{dx:+d}_dy{dy:+d}", dx=dx, dy=dy, kernel=kernel))
    return rules


def shift_mask(mask: Mask, dx: int, dy: int) -> Mask:
    height = len(mask)
    width = len(mask[0]) if height else 0
    out = [[False for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            tx = x + dx
            ty = y + dy
            if 0 <= tx < width and 0 <= ty < height:
                out[ty][tx] = True
    return out


def dilate_mask(mask: Mask, kernel: str) -> Mask:
    if kernel == "none":
        return [row[:] for row in mask]
    offsets_by_kernel = {
        "right_down": ((0, 0), (1, 0), (0, 1)),
        "cardinal": ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
        "box": (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ),
    }
    if kernel not in offsets_by_kernel:
        raise ValueError(f"unknown dilation kernel: {kernel}")
    height = len(mask)
    width = len(mask[0]) if height else 0
    out = [[False for _x in range(width)] for _y in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            for dx, dy in offsets_by_kernel[kernel]:
                tx = x + dx
                ty = y + dy
                if 0 <= tx < width and 0 <= ty < height:
                    out[ty][tx] = True
    return out


def apply_weight_rule(mask: Mask, rule: WeightRule) -> Mask:
    return shift_mask(dilate_mask(mask, rule.kernel), rule.dx, rule.dy)


def mask_to_levels(mask: Mask) -> list[list[int]]:
    return [[3 if value else 0 for value in row] for row in mask]


def _mean(records: list[BinaryMetrics], attr: str) -> float:
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def score_weight_rule(glyphs: list[dict], rule: WeightRule, *, limit: int | None = None) -> WeightRuleSummary:
    selected = glyphs[:limit] if limit is not None else glyphs
    records: list[BinaryMetrics] = []
    for glyph in selected:
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        records.append(compare_masks(apply_weight_rule(source_mask, rule), target_mask))
    return WeightRuleSummary(
        name=rule.name,
        dx=rule.dx,
        dy=rule.dy,
        kernel=rule.kernel,
        glyph_count=len(selected),
        foreground_f1=_mean(records, "foreground_f1"),
        foreground_iou=_mean(records, "foreground_iou"),
        foreground_precision=_mean(records, "foreground_precision"),
        foreground_recall=_mean(records, "foreground_recall"),
        pixel_accuracy=_mean(records, "pixel_accuracy"),
    )


def export_weight_search(
    source_metadata: Path = Path("data/processed/glyphs/source_metadata.json"),
    *,
    source_out_dir: Path = Path("data/processed/glyphs/source_weighted"),
    shadow_out_dir: Path = Path("data/processed/glyphs/baseline_weighted_shadow"),
    metadata_json: Path = Path("data/processed/glyphs/weight_search_metadata.json"),
    search_json: Path = Path("data/processed/glyphs/weight_search.json"),
    contact_sheet: Path = Path("data/processed/glyphs/baseline_weighted_shadow_contact.png"),
    search_limit: int | None = 512,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> WeightSearchExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    source_out_dir.mkdir(parents=True, exist_ok=True)
    shadow_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    search_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    summaries = [score_weight_rule(glyphs, rule, limit=search_limit) for rule in default_weight_rules()]
    summaries.sort(key=lambda item: (item.foreground_f1, item.foreground_iou), reverse=True)
    rule_by_name = {rule.name: rule for rule in default_weight_rules()}
    best_rule = rule_by_name[summaries[0].name]
    shadow_rule = ShadowRule("left_down_strong_diag_light", ((-1, 0, 2), (0, 1, 2), (-1, 1, 1)))

    source_binary_metrics: list[BinaryMetrics] = []
    shadow_visual_scores: list[float] = []
    shadow_records: list[BaselineGlyphMetrics] = []
    glyph_payloads: list[dict] = []
    for glyph in glyphs:
        index = int(glyph["index"])
        source_mask = image_to_mask(glyph["source_png"])
        target_mask = image_to_mask(glyph["target_png"])
        weighted_mask = apply_weight_rule(source_mask, best_rule)
        weighted_png = source_out_dir / f"glyph_{index:04d}.png"
        mask_to_image(weighted_mask).save(weighted_png)

        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        weighted_levels = mask_to_levels(weighted_mask)
        predicted_levels = apply_shadow_rule(weighted_levels, shadow_rule)
        predicted_png = shadow_out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)

        binary_metrics = compare_masks(weighted_mask, target_mask)
        visual_metrics = compare_visual(predicted_levels, target_levels)
        source_binary_metrics.append(binary_metrics)
        shadow_visual_scores.append(visual_metrics.visual_score)
        shadow_records.append(
            BaselineGlyphMetrics(
                index=index,
                codes=list(glyph["codes"]),
                chars=list(glyph["chars"]),
                source_png=str(weighted_png),
                predicted_png=str(predicted_png),
                target_png=str(glyph["target_png"]),
                pixel_accuracy=visual_metrics.pixel_accuracy,
                mean_absolute_error=visual_metrics.mean_absolute_error,
                foreground_iou=visual_metrics.foreground_iou,
            )
        )
        glyph_payloads.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": list(glyph["chars"]),
                "weighted_source_png": str(weighted_png),
                "predicted_png": str(predicted_png),
                "target_png": str(glyph["target_png"]),
                "binary": asdict(binary_metrics),
                "visual": visual_metrics.to_dict(),
            }
        )

    contact = make_baseline_contact_sheet(
        shadow_records,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    contact.save(contact_sheet)

    shadow_count = len(shadow_records)
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(glyphs),
        "cell_width": cell_width,
        "cell_height": cell_height,
        "best_weight_rule": asdict(summaries[0]),
        "shadow_rule": shadow_rule.name,
        "source_binary_summary": {
            "foreground_f1": _mean(source_binary_metrics, "foreground_f1"),
            "foreground_iou": _mean(source_binary_metrics, "foreground_iou"),
            "foreground_precision": _mean(source_binary_metrics, "foreground_precision"),
            "foreground_recall": _mean(source_binary_metrics, "foreground_recall"),
            "pixel_accuracy": _mean(source_binary_metrics, "pixel_accuracy"),
        },
        "shadow_summary": {
            "visual_score": sum(shadow_visual_scores) / shadow_count,
            "foreground_iou": sum(record.foreground_iou for record in shadow_records) / shadow_count,
            "pixel_accuracy": sum(record.pixel_accuracy for record in shadow_records) / shadow_count,
            "mean_absolute_error": sum(record.mean_absolute_error for record in shadow_records) / shadow_count,
        },
        "source_out_dir": str(source_out_dir),
        "shadow_out_dir": str(shadow_out_dir),
        "contact_sheet": str(contact_sheet),
        "glyphs": glyph_payloads,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    search_json.write_text(
        json.dumps(
            {
                "source_metadata": str(source_metadata),
                "search_limit": search_limit,
                "rules": [asdict(summary) for summary in summaries],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return WeightSearchExport(
        source_metadata=str(source_metadata),
        source_out_dir=str(source_out_dir),
        shadow_out_dir=str(shadow_out_dir),
        metadata_json=str(metadata_json),
        search_json=str(search_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(glyphs),
        best_rule=best_rule.name,
        source_foreground_f1=payload["source_binary_summary"]["foreground_f1"],
        source_foreground_iou=payload["source_binary_summary"]["foreground_iou"],
        shadow_visual_score=payload["shadow_summary"]["visual_score"],
        shadow_foreground_iou=payload["shadow_summary"]["foreground_iou"],
        shadow_pixel_accuracy=payload["shadow_summary"]["pixel_accuracy"],
        shadow_mean_absolute_error=payload["shadow_summary"]["mean_absolute_error"],
    )
