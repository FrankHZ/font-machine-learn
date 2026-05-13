from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from font_machine_learn.baseline import (
    BaselineGlyphMetrics,
    compare_levels,
    image_to_target_levels,
    levels_to_image,
    make_baseline_contact_sheet,
    source_image_to_levels,
)
from font_machine_learn.paths import BASELINE_MLP_CONTACT, BASELINE_MLP_DIR, BASELINE_MLP_METADATA, SOURCE_METADATA
from font_machine_learn.source_font import export_source_dataset


@dataclass(frozen=True)
class TrainableBaselineExport:
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    train_glyph_count: int
    heldout_glyph_count: int
    pixel_train_accuracy: float
    mean_pixel_accuracy: float
    mean_absolute_error: float
    mean_foreground_iou: float
    heldout_mean_pixel_accuracy: float
    heldout_mean_absolute_error: float
    heldout_mean_foreground_iou: float


def pixel_features(source_levels: list[list[int]], x: int, y: int) -> list[float]:
    height = len(source_levels)
    width = len(source_levels[0]) if height else 0
    values: list[float] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sx = x + dx
            sy = y + dy
            values.append(1.0 if 0 <= sx < width and 0 <= sy < height and source_levels[sy][sx] else 0.0)

    nx = x / max(1, width - 1)
    ny = y / max(1, height - 1)
    values.extend((nx, ny, min(nx, 1.0 - nx), min(ny, 1.0 - ny)))
    return values


def glyph_training_rows(
    source_levels: list[list[int]],
    target_levels: list[list[int]],
) -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for y, row in enumerate(target_levels):
        for x, value in enumerate(row):
            rows.append(pixel_features(source_levels, x, y))
            labels.append(value)
    return rows, labels


def levels_from_predictions(predictions: np.ndarray, width: int, height: int) -> list[list[int]]:
    values = [int(value) for value in predictions.tolist()]
    return [values[row * width : (row + 1) * width] for row in range(height)]


def mean_metric(records: list[BaselineGlyphMetrics], attr: str) -> float:
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def export_mlp_baseline(
    source_metadata: Path = SOURCE_METADATA,
    out_dir: Path = BASELINE_MLP_DIR,
    *,
    metadata_json: Path | None = BASELINE_MLP_METADATA,
    contact_sheet: Path | None = BASELINE_MLP_CONTACT,
    max_train_glyphs: int = 512,
    hidden_units: int = 48,
    max_iter: int = 80,
    random_seed: int = 13,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> TrainableBaselineExport:
    if not source_metadata.exists():
        export_source_dataset()

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = source["glyphs"]
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    train_glyph_count = min(max_train_glyphs, len(glyphs))

    train_rows: list[list[float]] = []
    train_labels: list[int] = []
    for glyph in glyphs[:train_glyph_count]:
        source_levels = source_image_to_levels(Image.open(glyph["source_png"]).convert("RGBA"))
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        rows, labels = glyph_training_rows(source_levels, target_levels)
        train_rows.extend(rows)
        train_labels.extend(labels)

    x_train = np.asarray(train_rows, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.int64)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        n_iter_no_change=8,
    )
    model.fit(x_train, y_train)
    pixel_train_accuracy = float(accuracy_score(y_train, model.predict(x_train)))

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_json or out_dir.parent / "baseline_mlp_metadata.json"
    contact_path = contact_sheet or out_dir.parent / "baseline_mlp_contact.png"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[BaselineGlyphMetrics] = []
    for glyph in glyphs:
        index = int(glyph["index"])
        source_levels = source_image_to_levels(Image.open(glyph["source_png"]).convert("RGBA"))
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        features = np.asarray(
            [pixel_features(source_levels, x, y) for y in range(cell_height) for x in range(cell_width)],
            dtype=np.float32,
        )
        predicted_levels = levels_from_predictions(model.predict(features), cell_width, cell_height)
        predicted_png = out_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        accuracy, mae, iou = compare_levels(predicted_levels, target_levels)
        records.append(
            BaselineGlyphMetrics(
                index=index,
                codes=list(glyph["codes"]),
                chars=list(glyph["chars"]),
                source_png=str(glyph["source_png"]),
                predicted_png=str(predicted_png),
                target_png=str(glyph["target_png"]),
                pixel_accuracy=accuracy,
                mean_absolute_error=mae,
                foreground_iou=iou,
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

    heldout = records[train_glyph_count:] or records
    payload = {
        "source_metadata": str(source_metadata),
        "glyph_count": len(records),
        "train_glyph_count": train_glyph_count,
        "heldout_glyph_count": len(glyphs) - train_glyph_count,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "model": {
            "kind": "sklearn.neural_network.MLPClassifier",
            "features": "3x3 source patch + normalized x/y + edge distances",
            "hidden_units": hidden_units,
            "max_iter": max_iter,
            "random_seed": random_seed,
            "n_iter": int(model.n_iter_),
        },
        "pixel_train_accuracy": pixel_train_accuracy,
        "mean_pixel_accuracy": mean_metric(records, "pixel_accuracy"),
        "mean_absolute_error": mean_metric(records, "mean_absolute_error"),
        "mean_foreground_iou": mean_metric(records, "foreground_iou"),
        "heldout_mean_pixel_accuracy": mean_metric(heldout, "pixel_accuracy"),
        "heldout_mean_absolute_error": mean_metric(heldout, "mean_absolute_error"),
        "heldout_mean_foreground_iou": mean_metric(heldout, "foreground_iou"),
        "glyph_dir": str(out_dir),
        "contact_sheet": str(contact_path),
        "glyphs": [asdict(record) for record in records],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return TrainableBaselineExport(
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_path),
        contact_sheet=str(contact_path),
        glyph_count=len(records),
        train_glyph_count=train_glyph_count,
        heldout_glyph_count=len(glyphs) - train_glyph_count,
        pixel_train_accuracy=pixel_train_accuracy,
        mean_pixel_accuracy=payload["mean_pixel_accuracy"],
        mean_absolute_error=payload["mean_absolute_error"],
        mean_foreground_iou=payload["mean_foreground_iou"],
        heldout_mean_pixel_accuracy=payload["heldout_mean_pixel_accuracy"],
        heldout_mean_absolute_error=payload["heldout_mean_absolute_error"],
        heldout_mean_foreground_iou=payload["heldout_mean_foreground_iou"],
    )
