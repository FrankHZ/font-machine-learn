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
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask, mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name
from font_machine_learn.external_eval import summarize_binary, summarize_visual, train_two_head_models
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.patch_classifier import model_n_iter, patch_features
from font_machine_learn.paths import (
    SONG13_ADAPTER_CONTACT,
    SONG13_ADAPTER_ERROR_CONTACT,
    SONG13_ADAPTER_METADATA,
    STAGE22_SONG13_ADAPTER,
    TARGET_METADATA,
)
from font_machine_learn.shadow_classifier import (
    binary_accuracy,
    binary_f1_from_confusion,
    merge_binary_confusion,
    predict_combined_levels,
    shadow_confusion,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


Mask = list[list[bool]]

DEFAULT_SONG13_SOURCE_METADATA = Path("data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json")


@dataclass(frozen=True)
class Song13AdapterExport:
    target_metadata: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    adapter_train_pixel_count: int
    best_adapter_model: str
    best_cjk_visual_score: float
    best_adapted_cjk_f1: float


def adapter_features(mask: Mask, x: int, y: int, radius: int) -> list[float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    x_scale = max(1, width - 1)
    y_scale = max(1, height - 1)
    return [
        *patch_features(mask, x, y, radius),
        x / x_scale,
        y / y_scale,
        min(x, width - 1 - x) / x_scale,
        min(y, height - 1 - y) / y_scale,
    ]


def build_adapter_training_rows(
    source_glyphs: list[dict],
    *,
    patch_radius: int,
    max_train_glyphs: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows: list[list[float]] = []
    labels: list[int] = []
    cjk_seen = 0
    for glyph in source_glyphs:
        if classify_glyph(list(glyph["chars"])) != "cjk":
            continue
        if max_train_glyphs is not None and cjk_seen >= max_train_glyphs:
            break
        cjk_seen += 1
        source_mask = image_to_mask(glyph["source_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_ge2 = levels_to_threshold_mask(target_levels, "ge2")
        for y, row in enumerate(source_mask):
            for x, _value in enumerate(row):
                rows.append(adapter_features(source_mask, x, y, patch_radius))
                labels.append(1 if target_ge2[y][x] else 0)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), cjk_seen


def balanced_adapter_subset(x_train: np.ndarray, y_train: np.ndarray, *, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y_train == 1)
    negative = np.flatnonzero(y_train == 0)
    if len(positive) == 0 or len(negative) == 0:
        return x_train, y_train
    rng = np.random.default_rng(random_seed)
    negative_count = min(len(negative), len(positive) * 2)
    selected_negative = rng.choice(negative, size=negative_count, replace=False)
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    return x_train[selected], y_train[selected]


def train_adapter_models(
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
    models["adapter_logistic_balanced"] = logistic

    x_balanced, y_balanced = balanced_adapter_subset(x_train, y_train, random_seed=random_seed)
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
        mlp.fit(x_balanced, y_balanced)
    models["adapter_patch_mlp"] = mlp
    return models


def predict_adapter_mask(model: object, source_mask: Mask, *, patch_radius: int) -> Mask:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    positions: list[tuple[int, int]] = []
    rows: list[list[float]] = []
    for y in range(height):
        for x in range(width):
            positions.append((x, y))
            rows.append(adapter_features(source_mask, x, y, patch_radius))
    predicted = [[False for _x in range(width)] for _y in range(height)]
    if rows:
        predictions = model.predict(np.asarray(rows, dtype=np.float32))
        for (x, y), value in zip(positions, predictions):
            predicted[y][x] = int(value) == 1
    return predicted


def evaluate_adapter_model(
    model_name: str,
    model: object,
    source_glyphs: list[dict],
    *,
    out_dir: Path,
    core_model: object,
    shadow_model: object,
    adapter_patch_radius: int,
    style_patch_radius: int,
) -> tuple[dict, list[BaselineGlyphMetrics], dict[int, float]]:
    adapted_dir = out_dir / model_name / "adapted_ge2"
    pred_dir = out_dir / model_name / "predicted_2bpp"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    source_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    adapted_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    shadow_matrix = {str(actual): {str(value): 0 for value in (0, 1)} for actual in (0, 1)}
    records: list[BaselineGlyphMetrics] = []
    glyph_payloads: list[dict] = []
    visual_by_index: dict[int, float] = {}

    for glyph in source_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = group_name(char_class)
        source_mask = image_to_mask(glyph["source_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_ge2 = levels_to_threshold_mask(target_levels, "ge2")
        adapted_mask = predict_adapter_mask(model, source_mask, patch_radius=adapter_patch_radius)

        adapted_png = adapted_dir / f"glyph_{index:04d}.png"
        predicted_png = pred_dir / f"glyph_{index:04d}.png"
        mask_to_image(adapted_mask).save(adapted_png)
        predicted_levels = predict_combined_levels(
            core_model,
            shadow_model,
            adapted_mask,
            patch_radius=style_patch_radius,
        )
        levels_to_image(predicted_levels).save(predicted_png)

        source_mask_metrics = compare_masks(source_mask, target_ge2)
        adapted_mask_metrics = compare_masks(adapted_mask, target_ge2)
        visual = compare_visual(predicted_levels, target_levels)
        visual_by_index[index] = visual.visual_score
        for key in ("all", group):
            visual_by_group[key].append(visual)
            source_mask_by_group[key].append(source_mask_metrics)
            adapted_mask_by_group[key].append(adapted_mask_metrics)
        if char_class == "cjk":
            merge_binary_confusion(shadow_matrix, shadow_confusion(predicted_levels, target_levels))

        record = BaselineGlyphMetrics(
            index=index,
            codes=list(glyph["codes"]),
            chars=chars,
            source_png=str(adapted_png),
            predicted_png=str(predicted_png),
            target_png=str(glyph["target_png"]),
            pixel_accuracy=visual.pixel_accuracy,
            mean_absolute_error=visual.mean_absolute_error,
            foreground_iou=visual.foreground_iou,
        )
        records.append(record)
        glyph_payloads.append(
            {
                **asdict(record),
                "original_source_png": str(glyph["source_png"]),
                "char_class": char_class,
                "source_mask_vs_target_ge2": asdict(source_mask_metrics),
                "adapted_mask_vs_target_ge2": asdict(adapted_mask_metrics),
                "visual": visual.to_dict(),
            }
        )

    payload = {
        "model": {
            "kind": type(model).__name__,
            "features": f"{adapter_patch_radius * 2 + 1}x{adapter_patch_radius * 2 + 1} source patch plus normalized coordinates",
            "n_iter": model_n_iter(model),
        },
        "adapted_dir": str(adapted_dir),
        "predicted_dir": str(pred_dir),
        "source_mask_vs_target_ge2": {
            key: summarize_binary(source_mask_by_group[key])
            for key in ("all", "cjk", "non_cjk")
        },
        "adapted_mask_vs_target_ge2": {
            key: summarize_binary(adapted_mask_by_group[key])
            for key in ("all", "cjk", "non_cjk")
        },
        "groups": {
            key: summarize_visual(visual_by_group[key])
            for key in ("all", "cjk", "non_cjk")
        },
        "shadow_cjk_confusion": {
            "accuracy": binary_accuracy(shadow_matrix),
            "foreground_f1": binary_f1_from_confusion(shadow_matrix),
            "matrix": shadow_matrix,
        },
        "glyphs": glyph_payloads,
    }
    return payload, records, visual_by_index


def export_song13_adapter(
    target_metadata: Path = TARGET_METADATA,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    *,
    out_dir: Path = STAGE22_SONG13_ADAPTER,
    metadata_json: Path = SONG13_ADAPTER_METADATA,
    contact_sheet: Path = SONG13_ADAPTER_CONTACT,
    error_contact_sheet: Path = SONG13_ADAPTER_ERROR_CONTACT,
    adapter_patch_radius: int = 3,
    style_patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    adapter_hidden_units: int = 64,
    style_core_hidden_units: int = 64,
    style_shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 22,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13AdapterExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    if not source_metadata.exists():
        raise FileNotFoundError(
            f"Source metadata not found: {source_metadata}. "
            "Render the Song13 baseline with scripts/render_source_glyphs.py first."
        )

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    target_glyphs = list(target["glyphs"])
    source_glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    core_model, shadow_model, style_train_payload = train_two_head_models(
        target_glyphs,
        patch_radius=style_patch_radius,
        max_train_glyphs=max_train_glyphs,
        core_hidden_units=style_core_hidden_units,
        shadow_hidden_units=style_shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )
    x_adapter, y_adapter, adapter_train_glyph_count = build_adapter_training_rows(
        source_glyphs,
        patch_radius=adapter_patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    adapter_models = train_adapter_models(
        x_adapter,
        y_adapter,
        hidden_units=adapter_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )

    model_payloads: dict[str, dict] = {}
    records_by_model: dict[str, list[BaselineGlyphMetrics]] = {}
    visual_by_index_by_model: dict[str, dict[int, float]] = {}
    for model_name, model in adapter_models.items():
        model_payload, records, visual_by_index = evaluate_adapter_model(
            model_name,
            model,
            source_glyphs,
            out_dir=out_dir,
            core_model=core_model,
            shadow_model=shadow_model,
            adapter_patch_radius=adapter_patch_radius,
            style_patch_radius=style_patch_radius,
        )
        model_payload["train_pixel_accuracy"] = float(accuracy_score(y_adapter, model.predict(x_adapter)))
        model_payloads[model_name] = model_payload
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
        "source_metadata": str(source_metadata),
        "glyph_count": len(source_glyphs),
        "cjk_glyph_count": len(best_records),
        "task": "adapt current Song13 1bpp source mask to target ge2 mask, then apply Stage20 two-head style model",
        "source_contract": "WenQuanYi Bitmap Song 13px, rendered at font-size 15 and left-bottom aligned",
        "target_adapter_label": "target level >= 2",
        "adapter_patch_radius": adapter_patch_radius,
        "adapter_patch_size": adapter_patch_radius * 2 + 1,
        "style_patch_radius": style_patch_radius,
        "style_patch_size": style_patch_radius * 2 + 1,
        "train_cjk_glyph_count": {
            "adapter": adapter_train_glyph_count,
            **style_train_payload["train_cjk_glyph_count"],
        },
        "train_pixel_count": {
            "adapter": int(len(y_adapter)),
            **style_train_payload["train_pixel_count"],
        },
        "label_counts": {
            "adapter": {str(value): int(np.count_nonzero(y_adapter == value)) for value in (0, 1)},
            **style_train_payload["label_counts"],
        },
        "style_model": {
            "core_edge": "patch_mlp trained on target-derived ge2 level 2/3 pixels",
            "shadow": "shadow_patch_mlp trained on target-derived ge2 outside pixels",
        },
        "best_adapter_model": best_model,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_model": best_model,
        "contact_sheet_order": ["adapted_ge2", "predicted_2bpp", "target_2bpp"],
        "models": model_payloads,
        "interpretation_notes": [
            "Stage21 measured the raw Song13 mask transfer bottleneck.",
            "Stage22 learns only a local source-mask adapter before the existing Stage20 style heads.",
            "If adapted_mask_vs_target_ge2 improves but visual remains low, the style heads are sensitive to adapter artifacts.",
            "If both improve, source adaptation is the right next axis before larger style models.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return Song13AdapterExport(
        target_metadata=str(target_metadata),
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(source_glyphs),
        cjk_glyph_count=len(best_records),
        adapter_train_pixel_count=int(len(y_adapter)),
        best_adapter_model=best_model,
        best_cjk_visual_score=float(model_payloads[best_model]["groups"]["cjk"]["visual_score"]),
        best_adapted_cjk_f1=float(model_payloads[best_model]["adapted_mask_vs_target_ge2"]["cjk"]["foreground_f1"]),
    )
