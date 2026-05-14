from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    PATCH_CLASSIFIER_CONTACT,
    PATCH_CLASSIFIER_ERROR_CONTACT,
    PATCH_CLASSIFIER_METADATA,
    STAGE19_PATCH_CLASSIFIER,
    TARGET_METADATA,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


Mask = list[list[bool]]
LevelGrid = list[list[int]]


@dataclass(frozen=True)
class PatchClassifierExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    train_pixel_count: int
    best_model: str
    best_cjk_visual_score: float
    best_ge2_accuracy: float


def patch_features(mask: Mask, x: int, y: int, radius: int) -> list[float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    values: list[float] = []
    for ty in range(y - radius, y + radius + 1):
        for tx in range(x - radius, x + radius + 1):
            values.append(1.0 if 0 <= ty < height and 0 <= tx < width and mask[ty][tx] else 0.0)
    return values


def build_training_rows(glyphs: list[dict], *, patch_radius: int, max_train_glyphs: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    rows: list[list[float]] = []
    labels: list[int] = []
    cjk_seen = 0
    for glyph in glyphs:
        if classify_glyph(list(glyph["chars"])) != "cjk":
            continue
        if max_train_glyphs is not None and cjk_seen >= max_train_glyphs:
            break
        cjk_seen += 1
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        source_mask = levels_to_threshold_mask(target_levels, "ge2")
        for y, row in enumerate(target_levels):
            for x, value in enumerate(row):
                if value not in (2, 3):
                    continue
                rows.append(patch_features(source_mask, x, y, patch_radius))
                labels.append(value)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), cjk_seen


def train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> dict[str, object]:
    models: dict[str, object] = {}
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=max(200, max_iter * 4),
        solver="lbfgs",
        random_state=random_seed,
    )
    logistic.fit(x_train, y_train)
    models["logistic_balanced"] = logistic

    mlp = MLPClassifier(
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(x_train, y_train)
    models["patch_mlp"] = mlp
    return models


def predict_ge2_levels(model: object, source_mask: Mask, *, patch_radius: int) -> LevelGrid:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]
    positions: list[tuple[int, int]] = []
    rows: list[list[float]] = []
    for y in range(height):
        for x in range(width):
            if source_mask[y][x]:
                positions.append((x, y))
                rows.append(patch_features(source_mask, x, y, patch_radius))
    if rows:
        predictions = model.predict(np.asarray(rows, dtype=np.float32))
        for (x, y), value in zip(positions, predictions):
            levels[y][x] = int(value)
    for y in range(height):
        for x in range(width):
            if levels[y][x] != 0:
                continue
            if 0 <= y - 1 < height and 0 <= x - 1 < width and source_mask[y - 1][x - 1]:
                levels[y][x] = 1
    return levels


def ge2_confusion(predicted: LevelGrid, target: LevelGrid) -> dict[str, dict[str, int]]:
    matrix = {str(actual): {str(value): 0 for value in range(4)} for actual in (2, 3)}
    for y, row in enumerate(target):
        for x, actual in enumerate(row):
            if actual not in (2, 3):
                continue
            matrix[str(actual)][str(predicted[y][x])] += 1
    return matrix


def merge_confusion(left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]) -> None:
    for actual, row in right.items():
        for predicted, count in row.items():
            left[actual][predicted] += count


def confusion_accuracy(matrix: dict[str, dict[str, int]]) -> float:
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[str(value)][str(value)] for value in (2, 3))
    return 0.0 if total == 0 else correct / total


def mean_visual(records: list[VisualMetrics], attr: str) -> float:
    return 0.0 if not records else sum(float(getattr(record, attr)) for record in records) / len(records)


def summarize_visual(records: list[VisualMetrics]) -> dict[str, float | int]:
    return {
        "glyph_count": len(records),
        "visual_score": mean_visual(records, "visual_score"),
        "ink_f1": mean_visual(records, "ink_f1"),
        "shadow_f1": mean_visual(records, "shadow_f1"),
        "foreground_iou": mean_visual(records, "foreground_iou"),
        "pixel_accuracy": mean_visual(records, "pixel_accuracy"),
        "mean_absolute_error": mean_visual(records, "mean_absolute_error"),
    }


def model_n_iter(model: object) -> int:
    value = getattr(model, "n_iter_", 0)
    if isinstance(value, np.ndarray):
        return int(value.max(initial=0))
    return int(value)


def export_patch_classifier(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE19_PATCH_CLASSIFIER,
    metadata_json: Path = PATCH_CLASSIFIER_METADATA,
    contact_sheet: Path = PATCH_CLASSIFIER_CONTACT,
    error_contact_sheet: Path = PATCH_CLASSIFIER_ERROR_CONTACT,
    patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 19,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> PatchClassifierExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_glyph_count = build_training_rows(
        glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    models = train_models(x_train, y_train, hidden_units=hidden_units, max_iter=max_iter, random_seed=random_seed)
    source_dir = out_dir / "source_ge2"
    source_dir.mkdir(parents=True, exist_ok=True)

    model_payloads: dict[str, dict] = {}
    records_by_model: dict[str, list[BaselineGlyphMetrics]] = {}
    visual_by_index_by_model: dict[str, dict[int, float]] = {}
    for model_name, model in models.items():
        pred_dir = out_dir / model_name / "predicted_2bpp"
        pred_dir.mkdir(parents=True, exist_ok=True)
        visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
        confusion = {str(actual): {str(value): 0 for value in range(4)} for actual in (2, 3)}
        records: list[BaselineGlyphMetrics] = []
        glyph_payloads: list[dict] = []
        visual_by_index: dict[int, float] = {}
        for glyph in glyphs:
            index = int(glyph["index"])
            chars = list(glyph["chars"])
            char_class = classify_glyph(chars)
            target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
            source_mask = levels_to_threshold_mask(target_levels, "ge2")
            source_png = source_dir / f"glyph_{index:04d}.png"
            if not source_png.exists():
                mask_to_image(source_mask).save(source_png)
            predicted_levels = predict_ge2_levels(model, source_mask, patch_radius=patch_radius)
            predicted_png = pred_dir / f"glyph_{index:04d}.png"
            levels_to_image(predicted_levels).save(predicted_png)
            visual = compare_visual(predicted_levels, target_levels)
            visual_by_index[index] = visual.visual_score
            for group in ("all", group_name(char_class)):
                visual_by_group[group].append(visual)
            if char_class == "cjk":
                merge_confusion(confusion, ge2_confusion(predicted_levels, target_levels))
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
            glyph_payloads.append({**asdict(record), "char_class": char_class, "visual": visual.to_dict()})

        train_accuracy = float(accuracy_score(y_train, model.predict(x_train)))
        groups = {name: summarize_visual(visual_by_group[name]) for name in ("all", "cjk", "non_cjk")}
        model_payloads[model_name] = {
            "model": {
                "kind": type(model).__name__,
                "features": f"{patch_radius * 2 + 1}x{patch_radius * 2 + 1} binary ge2 source patch only",
                "hidden_units": hidden_units if model_name == "patch_mlp" else None,
                "max_iter": max_iter,
                "random_seed": random_seed,
                "n_iter": model_n_iter(model),
            },
            "train_pixel_accuracy": train_accuracy,
            "ge2_cjk_confusion": {
                "accuracy": confusion_accuracy(confusion),
                "matrix": confusion,
            },
            "predicted_dir": str(pred_dir),
            "groups": groups,
            "glyphs": glyph_payloads,
        }
        records_by_model[model_name] = records
        visual_by_index_by_model[model_name] = visual_by_index

    best_model = max(model_payloads, key=lambda name: float(model_payloads[name]["groups"]["cjk"]["visual_score"]))
    best_records = [record for record in records_by_model[best_model] if classify_glyph(record.chars) == "cjk"]
    _make_contact_sheet(
        best_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst_records = sorted(best_records, key=lambda record: visual_by_index_by_model[best_model][record.index])[:worst_count]
    _make_contact_sheet(
        worst_records,
        out_path=error_contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(glyphs),
        "cjk_glyph_count": len(best_records),
        "task": "patch-only classifier for ge2 source pixels: target level 2 vs 3",
        "source_mask": "target-derived ge2 mask, value >= 2",
        "outside_ge2_rule": "right-down shadow level 1 from source ge2; otherwise level 0",
        "patch_radius": patch_radius,
        "patch_size": patch_radius * 2 + 1,
        "train_cjk_glyph_count": train_glyph_count,
        "train_pixel_count": int(len(y_train)),
        "label_counts": {str(value): int(np.count_nonzero(y_train == value)) for value in (2, 3)},
        "best_model": best_model,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_model": best_model,
        "contact_sheet_order": ["source_ge2", "predicted_2bpp", "target_2bpp"],
        "models": model_payloads,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return PatchClassifierExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(glyphs),
        cjk_glyph_count=len(best_records),
        train_pixel_count=int(len(y_train)),
        best_model=best_model,
        best_cjk_visual_score=float(model_payloads[best_model]["groups"]["cjk"]["visual_score"]),
        best_ge2_accuracy=float(model_payloads[best_model]["ge2_cjk_confusion"]["accuracy"]),
    )
