from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels
from font_machine_learn.nftr import checkerboard
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class ReviewBaseline:
    name: str
    metadata_json: Path


@dataclass(frozen=True)
class ReviewExport:
    source_metadata: str
    review_json: str
    worst_cases_json: str
    contact_sheet: str
    glyph_count: int
    selected_count: int
    baselines: list[str]


def _load_baseline_glyphs(spec: ReviewBaseline) -> dict[int, dict]:
    payload = json.loads(spec.metadata_json.read_text(encoding="utf-8"))
    return {int(glyph["index"]): glyph for glyph in payload["glyphs"]}


def _mean_metric(metrics: list[VisualMetrics], attr: str) -> float:
    return sum(float(getattr(metric, attr)) for metric in metrics) / len(metrics)


def _summarize_metrics(metrics: list[VisualMetrics]) -> dict[str, float]:
    return {
        "visual_score": _mean_metric(metrics, "visual_score"),
        "ink_f1": _mean_metric(metrics, "ink_f1"),
        "shadow_f1": _mean_metric(metrics, "shadow_f1"),
        "foreground_iou": _mean_metric(metrics, "foreground_iou"),
        "weighted_similarity": _mean_metric(metrics, "weighted_similarity"),
        "isolated_foreground_fraction": _mean_metric(metrics, "isolated_foreground_fraction"),
        "pixel_accuracy": _mean_metric(metrics, "pixel_accuracy"),
        "mean_absolute_error": _mean_metric(metrics, "mean_absolute_error"),
    }


def _make_review_contact_sheet(
    rows: list[dict],
    *,
    image_keys: list[str],
    out_path: Path,
    scale: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    pad: int,
) -> None:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    group_w = tile_w * len(image_keys) + pad * (len(image_keys) - 1)
    rows_per_sheet = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (
            columns * group_w + (columns + 1) * pad,
            rows_per_sheet * tile_h + (rows_per_sheet + 1) * pad,
        ),
        max(2, scale * 2),
    )

    for position, row in enumerate(rows):
        col = position % columns
        sheet_row = position // columns
        x = pad + col * (group_w + pad)
        y = pad + sheet_row * (tile_h + pad)
        for image_index, key in enumerate(image_keys):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x + image_index * (tile_w + pad), y))
    sheet.save(out_path)


def export_review_report(
    source_metadata: Path = Path("data/processed/glyphs/source_metadata.json"),
    baselines: list[ReviewBaseline] | None = None,
    *,
    review_json: Path = Path("data/processed/glyphs/review_report.json"),
    worst_cases_json: Path = Path("data/processed/glyphs/review_worst_cases.json"),
    contact_sheet: Path = Path("data/processed/glyphs/review_contact.png"),
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 8,
    pad: int = 1,
) -> ReviewExport:
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = source["glyphs"]
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    baseline_specs = baselines or [
        ReviewBaseline("shadow", Path("data/processed/glyphs/baseline_shadow_metadata.json")),
        ReviewBaseline("tuned", Path("data/processed/glyphs/baseline_tuned_shadow_metadata.json")),
        ReviewBaseline("mlp", Path("data/processed/glyphs/baseline_mlp_metadata.json")),
    ]

    loaded = {spec.name: _load_baseline_glyphs(spec) for spec in baseline_specs}
    metrics_by_baseline: dict[str, list[VisualMetrics]] = {spec.name: [] for spec in baseline_specs}
    rows: list[dict] = []
    for glyph in glyphs:
        index = int(glyph["index"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        row: dict[str, object] = {
            "index": index,
            "codes": list(glyph["codes"]),
            "chars": list(glyph["chars"]),
            "source_png": glyph["source_png"],
            "target_png": glyph["target_png"],
            "baselines": {},
        }
        worst_visual = 1.0
        for spec in baseline_specs:
            baseline_glyph = loaded[spec.name][index]
            predicted_png = baseline_glyph["predicted_png"]
            predicted_levels = image_to_target_levels(Image.open(predicted_png).convert("RGBA"))
            metrics = compare_visual(predicted_levels, target_levels)
            metrics_by_baseline[spec.name].append(metrics)
            row[f"{spec.name}_png"] = predicted_png
            row["baselines"][spec.name] = metrics.to_dict()  # type: ignore[index]
            worst_visual = min(worst_visual, metrics.visual_score)
        row["worst_visual_score"] = worst_visual
        rows.append(row)

    summaries = {
        name: _summarize_metrics(metrics)
        for name, metrics in metrics_by_baseline.items()
    }
    sorted_rows = sorted(rows, key=lambda row: float(row["worst_visual_score"]))
    selected = sorted_rows[:worst_count]
    image_keys = ["source_png"] + [f"{spec.name}_png" for spec in baseline_specs] + ["target_png"]

    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    review_json.parent.mkdir(parents=True, exist_ok=True)
    worst_cases_json.parent.mkdir(parents=True, exist_ok=True)
    _make_review_contact_sheet(
        selected,
        image_keys=image_keys,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(rows),
        "selected_count": len(selected),
        "image_order": ["source"] + [spec.name for spec in baseline_specs] + ["target"],
        "summaries": summaries,
        "contact_sheet": str(contact_sheet),
        "worst_cases_json": str(worst_cases_json),
    }
    worst_payload = {
        "sort": "ascending worst visual_score across compared baselines",
        "image_order": payload["image_order"],
        "glyphs": selected,
    }
    review_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    worst_cases_json.write_text(
        json.dumps(worst_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return ReviewExport(
        source_metadata=str(source_metadata),
        review_json=str(review_json),
        worst_cases_json=str(worst_cases_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(rows),
        selected_count=len(selected),
        baselines=[spec.name for spec in baseline_specs],
    )
