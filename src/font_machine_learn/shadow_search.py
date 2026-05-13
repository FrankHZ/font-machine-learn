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
from font_machine_learn.paths import (
    BASELINE_TUNED_SHADOW_CONTACT,
    BASELINE_TUNED_SHADOW_DIR,
    BASELINE_TUNED_SHADOW_METADATA,
    BASELINE_TUNED_SHADOW_SEARCH,
    SOURCE_METADATA,
)
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.visual_metrics import LevelGrid, VisualMetrics, compare_visual


@dataclass(frozen=True)
class ShadowRule:
    name: str
    offsets: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class RuleSummary:
    name: str
    offsets: list[tuple[int, int, int]]
    glyph_count: int
    mean_visual_score: float
    mean_ink_f1: float
    mean_shadow_f1: float
    mean_foreground_iou: float
    mean_weighted_similarity: float
    mean_isolated_foreground_fraction: float
    mean_pixel_accuracy: float
    mean_absolute_error: float


@dataclass(frozen=True)
class TunedShadowExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    search_json: str
    contact_sheet: str
    glyph_count: int
    best_rule: str
    mean_visual_score: float
    mean_ink_f1: float
    mean_shadow_f1: float
    mean_foreground_iou: float
    mean_pixel_accuracy: float
    mean_absolute_error: float


def default_shadow_rules() -> list[ShadowRule]:
    return [
        ShadowRule("right_down_strong_diag_light", ((1, 0, 2), (0, 1, 2), (1, 1, 1))),
        ShadowRule("right_down_all_mid", ((1, 0, 2), (0, 1, 2), (1, 1, 2))),
        ShadowRule("right_down_light", ((1, 0, 1), (0, 1, 1), (1, 1, 1))),
        ShadowRule("right_down_no_diag", ((1, 0, 2), (0, 1, 2))),
        ShadowRule("down_right_diag_only", ((1, 1, 2),)),
        ShadowRule("right_only_mid", ((1, 0, 2),)),
        ShadowRule("down_only_mid", ((0, 1, 2),)),
        ShadowRule("left_down_strong_diag_light", ((-1, 0, 2), (0, 1, 2), (-1, 1, 1))),
        ShadowRule("right_up_strong_diag_light", ((1, 0, 2), (0, -1, 2), (1, -1, 1))),
        ShadowRule("left_up_strong_diag_light", ((-1, 0, 2), (0, -1, 2), (-1, -1, 1))),
        ShadowRule("down_heavy_with_diag", ((0, 1, 2), (1, 1, 1), (-1, 1, 1))),
        ShadowRule("right_heavy_with_diag", ((1, 0, 2), (1, 1, 1), (1, -1, 1))),
    ]


def apply_shadow_rule(source_levels: LevelGrid, rule: ShadowRule) -> LevelGrid:
    height = len(source_levels)
    width = len(source_levels[0]) if height else 0
    out = [row[:] for row in source_levels]
    for y in range(height):
        for x in range(width):
            if source_levels[y][x] != 3:
                continue
            for dx, dy, value in rule.offsets:
                sx = x + dx
                sy = y + dy
                if 0 <= sx < width and 0 <= sy < height and out[sy][sx] < value:
                    out[sy][sx] = value
    return out


def _mean_metrics(metrics: list[VisualMetrics]) -> dict[str, float]:
    if not metrics:
        raise ValueError("at least one metric record is required")
    names = (
        "visual_score",
        "ink_f1",
        "shadow_f1",
        "foreground_iou",
        "weighted_similarity",
        "isolated_foreground_fraction",
        "pixel_accuracy",
        "mean_absolute_error",
    )
    return {
        name: sum(getattr(metric, name) for metric in metrics) / len(metrics)
        for name in names
    }


def score_rule(source_glyphs: list[dict], rule: ShadowRule, *, limit: int | None = None) -> RuleSummary:
    metrics: list[VisualMetrics] = []
    selected = source_glyphs[:limit] if limit is not None else source_glyphs
    for glyph in selected:
        source = Image.open(glyph["source_png"]).convert("RGBA")
        target = Image.open(glyph["target_png"]).convert("RGBA")
        predicted = apply_shadow_rule(source_image_to_levels(source), rule)
        metrics.append(compare_visual(predicted, image_to_target_levels(target)))

    means = _mean_metrics(metrics)
    return RuleSummary(
        name=rule.name,
        offsets=list(rule.offsets),
        glyph_count=len(selected),
        mean_visual_score=means["visual_score"],
        mean_ink_f1=means["ink_f1"],
        mean_shadow_f1=means["shadow_f1"],
        mean_foreground_iou=means["foreground_iou"],
        mean_weighted_similarity=means["weighted_similarity"],
        mean_isolated_foreground_fraction=means["isolated_foreground_fraction"],
        mean_pixel_accuracy=means["pixel_accuracy"],
        mean_absolute_error=means["mean_absolute_error"],
    )


def export_tuned_shadow_baseline(
    source_metadata: Path = SOURCE_METADATA,
    out_dir: Path = BASELINE_TUNED_SHADOW_DIR,
    *,
    metadata_json: Path | None = BASELINE_TUNED_SHADOW_METADATA,
    search_json: Path | None = BASELINE_TUNED_SHADOW_SEARCH,
    contact_sheet: Path | None = BASELINE_TUNED_SHADOW_CONTACT,
    search_limit: int | None = 512,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> TunedShadowExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_json or out_dir.parent / "baseline_tuned_shadow_metadata.json"
    search_path = search_json or out_dir.parent / "baseline_tuned_shadow_search.json"
    contact_path = contact_sheet or out_dir.parent / "baseline_tuned_shadow_contact.png"

    summaries = [score_rule(glyphs, rule, limit=search_limit) for rule in default_shadow_rules()]
    summaries.sort(key=lambda summary: summary.mean_visual_score, reverse=True)
    rule_by_name = {rule.name: rule for rule in default_shadow_rules()}
    best_rule = rule_by_name[summaries[0].name]

    records: list[BaselineGlyphMetrics] = []
    visual_records: list[dict[str, object]] = []
    visual_metrics: list[VisualMetrics] = []
    for glyph in glyphs:
        index = int(glyph["index"])
        source_png = str(glyph["source_png"])
        target_png = str(glyph["target_png"])
        source_image = Image.open(source_png).convert("RGBA")
        target_image = Image.open(target_png).convert("RGBA")
        predicted_levels = apply_shadow_rule(source_image_to_levels(source_image), best_rule)
        target_levels = image_to_target_levels(target_image)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        metrics = compare_visual(predicted_levels, target_levels)
        visual_metrics.append(metrics)
        visual_records.append({"index": index, **metrics.to_dict()})
        records.append(
            BaselineGlyphMetrics(
                index=index,
                codes=list(glyph["codes"]),
                chars=list(glyph["chars"]),
                source_png=source_png,
                predicted_png=str(predicted_png),
                target_png=target_png,
                pixel_accuracy=metrics.pixel_accuracy,
                mean_absolute_error=metrics.mean_absolute_error,
                foreground_iou=metrics.foreground_iou,
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

    means = _mean_metrics(visual_metrics)
    search_payload = {
        "source_metadata": str(source_metadata),
        "search_limit": search_limit,
        "rules": [asdict(summary) for summary in summaries],
    }
    search_path.write_text(json.dumps(search_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(records),
        "cell_width": cell_width,
        "cell_height": cell_height,
        "best_rule": asdict(summaries[0]),
        "all_rule_summaries": [asdict(summary) for summary in summaries],
        "means": means,
        "glyph_dir": str(out_dir),
        "contact_sheet": str(contact_path),
        "glyphs": [
            {**asdict(record), **visual_record}
            for record, visual_record in zip(records, visual_records)
        ],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return TunedShadowExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_path),
        search_json=str(search_path),
        contact_sheet=str(contact_path),
        glyph_count=len(records),
        best_rule=best_rule.name,
        mean_visual_score=means["visual_score"],
        mean_ink_f1=means["ink_f1"],
        mean_shadow_f1=means["shadow_f1"],
        mean_foreground_iou=means["foreground_iou"],
        mean_pixel_accuracy=means["pixel_accuracy"],
        mean_absolute_error=means["mean_absolute_error"],
    )
