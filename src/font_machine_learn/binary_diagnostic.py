from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels
from font_machine_learn.nftr import checkerboard


@dataclass(frozen=True)
class BinaryMetrics:
    pixel_accuracy: float
    foreground_precision: float
    foreground_recall: float
    foreground_f1: float
    foreground_iou: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class BinaryDiagnosticExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    baselines: list[str]


def levels_to_mask(levels: list[list[int]]) -> list[list[bool]]:
    return [[value > 0 for value in row] for row in levels]


def mask_to_image(mask: list[list[bool]]) -> Image.Image:
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("RGBA", (width, height))
    image.putdata(
        [
            (0, 0, 0, 255) if value else (0, 0, 0, 0)
            for row in mask
            for value in row
        ]
    )
    return image


def image_to_mask(path: str | Path) -> list[list[bool]]:
    return levels_to_mask(image_to_target_levels(Image.open(path).convert("RGBA")))


def flatten_mask(mask: list[list[bool]]) -> list[bool]:
    return [value for row in mask for value in row]


def compare_masks(predicted_mask: list[list[bool]], target_mask: list[list[bool]]) -> BinaryMetrics:
    predicted = flatten_mask(predicted_mask)
    target = flatten_mask(target_mask)
    if len(predicted) != len(target):
        raise ValueError("predicted and target masks have different sizes")

    true_positive = sum(1 for left, right in zip(predicted, target) if left and right)
    true_negative = sum(1 for left, right in zip(predicted, target) if not left and not right)
    false_positive = sum(1 for left, right in zip(predicted, target) if left and not right)
    false_negative = sum(1 for left, right in zip(predicted, target) if not left and right)
    total = len(target)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    precision = 1.0 if precision_denominator == 0 else true_positive / precision_denominator
    recall = 1.0 if recall_denominator == 0 else true_positive / recall_denominator
    f1_denominator = precision + recall
    f1 = 1.0 if f1_denominator == 0 else 2 * precision * recall / f1_denominator
    iou = 1.0 if iou_denominator == 0 else true_positive / iou_denominator
    negatives = true_negative + false_positive
    positives = true_positive + false_negative
    return BinaryMetrics(
        pixel_accuracy=(true_positive + true_negative) / total,
        foreground_precision=precision,
        foreground_recall=recall,
        foreground_f1=f1,
        foreground_iou=iou,
        false_positive_rate=0.0 if negatives == 0 else false_positive / negatives,
        false_negative_rate=0.0 if positives == 0 else false_negative / positives,
    )


def _mean(records: list[BinaryMetrics], attr: str) -> float:
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def _summarize(records: list[BinaryMetrics]) -> dict[str, float]:
    return {
        "pixel_accuracy": _mean(records, "pixel_accuracy"),
        "foreground_precision": _mean(records, "foreground_precision"),
        "foreground_recall": _mean(records, "foreground_recall"),
        "foreground_f1": _mean(records, "foreground_f1"),
        "foreground_iou": _mean(records, "foreground_iou"),
        "false_positive_rate": _mean(records, "false_positive_rate"),
        "false_negative_rate": _mean(records, "false_negative_rate"),
    }


def _load_baseline(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(glyph["index"]): glyph for glyph in payload["glyphs"]}


def _make_contact_sheet(
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
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (
            columns * group_w + (columns + 1) * pad,
            sheet_rows * tile_h + (sheet_rows + 1) * pad,
        ),
        max(2, scale * 2),
    )
    for position, row in enumerate(rows):
        x = pad + (position % columns) * (group_w + pad)
        y = pad + (position // columns) * (tile_h + pad)
        for image_index, key in enumerate(image_keys):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x + image_index * (tile_w + pad), y))
    sheet.save(out_path)


def export_binary_diagnostic(
    source_metadata: Path = Path("data/processed/glyphs/source_metadata.json"),
    *,
    baseline_metadata: dict[str, Path] | None = None,
    out_dir: Path = Path("data/processed/glyphs/target_1bpp"),
    metadata_json: Path = Path("data/processed/glyphs/binary_diagnostic_metadata.json"),
    contact_sheet: Path = Path("data/processed/glyphs/binary_diagnostic_contact.png"),
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 8,
    pad: int = 1,
) -> BinaryDiagnosticExport:
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = source["glyphs"]
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    baselines = baseline_metadata or {
        "source": source_metadata,
        "shadow": Path("data/processed/glyphs/baseline_shadow_metadata.json"),
        "tuned": Path("data/processed/glyphs/baseline_tuned_shadow_metadata.json"),
        "mlp": Path("data/processed/glyphs/baseline_mlp_metadata.json"),
    }
    loaded_baselines = {
        name: None if path == source_metadata else _load_baseline(path)
        for name, path in baselines.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_baseline: dict[str, list[BinaryMetrics]] = {name: [] for name in baselines}
    rows: list[dict] = []

    for glyph in glyphs:
        index = int(glyph["index"])
        target_mask = image_to_mask(glyph["target_png"])
        target_binary_png = out_dir / f"glyph_{index:04d}.png"
        mask_to_image(target_mask).save(target_binary_png)
        row: dict[str, object] = {
            "index": index,
            "codes": list(glyph["codes"]),
            "chars": list(glyph["chars"]),
            "target_1bpp_png": str(target_binary_png),
            "baselines": {},
        }
        best_f1 = 0.0
        for name, baseline_glyphs in loaded_baselines.items():
            if baseline_glyphs is None:
                predicted_png = glyph["source_png"]
            else:
                predicted_png = baseline_glyphs[index]["predicted_png"]
            predicted_mask = image_to_mask(predicted_png)
            metrics = compare_masks(predicted_mask, target_mask)
            metrics_by_baseline[name].append(metrics)
            row[f"{name}_png"] = predicted_png
            row["baselines"][name] = asdict(metrics)  # type: ignore[index]
            best_f1 = max(best_f1, metrics.foreground_f1)
        row["best_foreground_f1"] = best_f1
        rows.append(row)

    sorted_rows = sorted(rows, key=lambda row: float(row["best_foreground_f1"]))
    selected = sorted_rows[:worst_count]
    image_keys = [f"{name}_png" for name in baselines] + ["target_1bpp_png"]
    _make_contact_sheet(
        selected,
        image_keys=image_keys,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    summaries = {
        name: _summarize(records)
        for name, records in metrics_by_baseline.items()
    }
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(glyphs),
        "cell_width": cell_width,
        "cell_height": cell_height,
        "target_1bpp_dir": str(out_dir),
        "image_order": list(baselines) + ["target_1bpp"],
        "summaries": summaries,
        "worst_count": len(selected),
        "contact_sheet": str(contact_sheet),
        "glyphs": rows,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return BinaryDiagnosticExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(glyphs),
        baselines=list(baselines),
    )
