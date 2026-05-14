from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import group_name
from font_machine_learn.external_eval import summarize_binary, summarize_visual, train_two_head_models
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    SONG13_CALIBRATED_CONTACT,
    SONG13_CALIBRATED_ERROR_CONTACT,
    SONG13_CALIBRATED_METADATA,
    STAGE23_SONG13_CALIBRATED,
    TARGET_METADATA,
)
from font_machine_learn.shadow_classifier import (
    binary_accuracy,
    binary_f1_from_confusion,
    merge_binary_confusion,
    predict_combined_levels,
    shadow_confusion,
)
from font_machine_learn.song13_adapter import (
    DEFAULT_SONG13_SOURCE_METADATA,
    Mask,
    adapter_features,
    build_adapter_training_rows,
    ge2_mask_to_image,
    make_adapter_contact_sheet,
    train_adapter_models,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class Song13CalibratedExport:
    target_metadata: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    adapter_train_pixel_count: int
    best_candidate: str
    best_adapter_model: str
    best_threshold: float
    best_cjk_quality_score: float
    best_cjk_visual_score: float
    best_adapted_cjk_f1: float
    best_adapted_cjk_foreground_ratio: float
    target_cjk_foreground_ratio: float


def positive_probabilities(model: object, rows: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(rows), dtype=np.float32)
    classes = list(getattr(model, "classes_", []))
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return np.asarray(model.predict_proba(rows)[:, positive_index], dtype=np.float32)


def adapter_feature_matrix(source_mask: Mask, *, patch_radius: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    positions: list[tuple[int, int]] = []
    rows: list[list[float]] = []
    for y, row in enumerate(source_mask):
        for x, _value in enumerate(row):
            positions.append((x, y))
            rows.append(adapter_features(source_mask, x, y, patch_radius))
    return np.asarray(rows, dtype=np.float32), positions


def threshold_mask(probabilities: np.ndarray, positions: list[tuple[int, int]], width: int, height: int, threshold: float) -> Mask:
    predicted = [[False for _x in range(width)] for _y in range(height)]
    for (x, y), probability in zip(positions, probabilities):
        predicted[y][x] = float(probability) >= threshold
    return predicted


def mask_foreground_ratio(mask: Mask) -> float:
    total = sum(len(row) for row in mask)
    if total == 0:
        return 0.0
    return sum(1 for row in mask for value in row if value) / total


def mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def quality_score(candidate: dict) -> float:
    groups = candidate["groups"]["cjk"]
    adapted = candidate["adapted_mask_vs_target_ge2"]["cjk"]
    ratios = candidate["foreground_ratio"]["cjk"]
    ratio_penalty = abs(float(ratios["adapted"]) - float(ratios["target_ge2"]))
    return (
        0.45 * float(groups["visual_score"])
        + 0.25 * float(adapted["foreground_f1"])
        + 0.15 * float(adapted["foreground_precision"])
        + 0.15 * float(groups["ink_f1"])
        - 0.90 * ratio_penalty
    )


def evaluate_candidate(
    candidate_name: str,
    model_name: str,
    model: object,
    threshold: float,
    source_glyphs: list[dict],
    *,
    out_dir: Path,
    core_model: object,
    shadow_model: object,
    adapter_patch_radius: int,
    style_patch_radius: int,
) -> tuple[dict, dict[int, float]]:
    adapted_dir = out_dir / candidate_name / "adapted_ge2"
    pred_dir = out_dir / candidate_name / "predicted_2bpp"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    source_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    adapted_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    ratios = {
        "all": {"source": [], "adapted": [], "target_ge2": []},
        "cjk": {"source": [], "adapted": [], "target_ge2": []},
        "non_cjk": {"source": [], "adapted": [], "target_ge2": []},
    }
    shadow_matrix = {str(actual): {str(value): 0 for value in (0, 1)} for actual in (0, 1)}
    glyph_payloads: list[dict] = []
    visual_by_index: dict[int, float] = {}

    for glyph in source_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = group_name(char_class)
        source_mask = image_to_mask(glyph["source_png"])
        height = len(source_mask)
        width = len(source_mask[0]) if height else 0
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_ge2 = levels_to_threshold_mask(target_levels, "ge2")
        rows, positions = adapter_feature_matrix(source_mask, patch_radius=adapter_patch_radius)
        probabilities = positive_probabilities(model, rows)
        adapted_mask = threshold_mask(probabilities, positions, width, height, threshold)

        adapted_png = adapted_dir / f"glyph_{index:04d}.png"
        predicted_png = pred_dir / f"glyph_{index:04d}.png"
        ge2_mask_to_image(adapted_mask).save(adapted_png)
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
            ratios[key]["source"].append(mask_foreground_ratio(source_mask))
            ratios[key]["adapted"].append(mask_foreground_ratio(adapted_mask))
            ratios[key]["target_ge2"].append(mask_foreground_ratio(target_ge2))
        if char_class == "cjk":
            merge_binary_confusion(shadow_matrix, shadow_confusion(predicted_levels, target_levels))

        glyph_payloads.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": chars,
                "original_source_png": str(glyph["source_png"]),
                "source_png": str(adapted_png),
                "predicted_png": str(predicted_png),
                "target_png": str(glyph["target_png"]),
                "char_class": char_class,
                "source_mask_vs_target_ge2": asdict(source_mask_metrics),
                "adapted_mask_vs_target_ge2": asdict(adapted_mask_metrics),
                "visual": visual.to_dict(),
                "foreground_ratio": {
                    "source": mask_foreground_ratio(source_mask),
                    "adapted": mask_foreground_ratio(adapted_mask),
                    "target_ge2": mask_foreground_ratio(target_ge2),
                },
            }
        )

    payload = {
        "adapter_model": model_name,
        "threshold": threshold,
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
        "foreground_ratio": {
            key: {
                "source": mean(ratios[key]["source"]),
                "adapted": mean(ratios[key]["adapted"]),
                "target_ge2": mean(ratios[key]["target_ge2"]),
                "adapted_minus_target_ge2": mean(ratios[key]["adapted"]) - mean(ratios[key]["target_ge2"]),
            }
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
    payload["cjk_quality_score"] = quality_score(payload)
    return payload, visual_by_index


def export_song13_calibrated(
    target_metadata: Path = TARGET_METADATA,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    *,
    out_dir: Path = STAGE23_SONG13_CALIBRATED,
    metadata_json: Path = SONG13_CALIBRATED_METADATA,
    contact_sheet: Path = SONG13_CALIBRATED_CONTACT,
    error_contact_sheet: Path = SONG13_CALIBRATED_ERROR_CONTACT,
    thresholds: list[float] | None = None,
    adapter_patch_radius: int = 3,
    style_patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    adapter_hidden_units: int = 64,
    style_core_hidden_units: int = 64,
    style_shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 23,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13CalibratedExport:
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
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

    candidate_payloads: dict[str, dict] = {}
    visual_by_index_by_candidate: dict[str, dict[int, float]] = {}
    for model_name, model in adapter_models.items():
        train_probability_accuracy = float(accuracy_score(y_adapter, positive_probabilities(model, x_adapter) >= 0.5))
        for threshold in thresholds:
            candidate_name = f"{model_name}_t{int(round(threshold * 100)):03d}"
            payload, visual_by_index = evaluate_candidate(
                candidate_name,
                model_name,
                model,
                float(threshold),
                source_glyphs,
                out_dir=out_dir,
                core_model=core_model,
                shadow_model=shadow_model,
                adapter_patch_radius=adapter_patch_radius,
                style_patch_radius=style_patch_radius,
            )
            payload["train_probability_accuracy_at_0_50"] = train_probability_accuracy
            candidate_payloads[candidate_name] = payload
            visual_by_index_by_candidate[candidate_name] = visual_by_index

    best_candidate = max(candidate_payloads, key=lambda name: float(candidate_payloads[name]["cjk_quality_score"]))
    best_payload = candidate_payloads[best_candidate]
    best_contact_records = [
        glyph
        for glyph in best_payload["glyphs"]
        if glyph["char_class"] == "cjk"
    ]
    make_adapter_contact_sheet(
        best_contact_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst_records = sorted(
        best_contact_records,
        key=lambda record: float(record["visual"]["visual_score"]),
    )[:worst_count]
    make_adapter_contact_sheet(
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
        "cjk_glyph_count": len(best_contact_records),
        "task": "calibrate Song13 source-mask adapter threshold before Stage20 style heads",
        "source_contract": "WenQuanYi Bitmap Song 13px, rendered at font-size 15 and left-bottom aligned",
        "target_adapter_label": "target level >= 2",
        "adapted_ge2_visualization": "foreground pixels are saved as NFTR level 2 gray for readability; metrics treat this as a binary ge2 mask",
        "selection": {
            "primary": "cjk_quality_score",
            "formula": "0.45*visual + 0.25*adapted_f1 + 0.15*adapted_precision + 0.15*ink_f1 - 0.90*abs(adapted_ratio-target_ge2_ratio)",
            "reason": "Stage22 looked too heavy by eye; Stage23 penalizes over-inking while keeping precision in the score.",
        },
        "thresholds": thresholds,
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
        "best_candidate": best_candidate,
        "best_adapter_model": best_payload["adapter_model"],
        "best_threshold": best_payload["threshold"],
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_candidate": best_candidate,
        "contact_sheet_order": ["original_source", "adapted_ge2", "predicted_2bpp", "target_2bpp"],
        "candidates": candidate_payloads,
        "interpretation_notes": [
            "Stage22 proved source adaptation has signal but default thresholding over-inks dense CJK glyphs.",
            "Stage23 keeps the same model family and calibrates threshold selection with an explicit ink-ratio penalty.",
            "If this improves contact sheets but lowers raw visual score, trust the contact sheet for the next source-adapter decision.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return Song13CalibratedExport(
        target_metadata=str(target_metadata),
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(source_glyphs),
        cjk_glyph_count=len(best_contact_records),
        adapter_train_pixel_count=int(len(y_adapter)),
        best_candidate=best_candidate,
        best_adapter_model=str(best_payload["adapter_model"]),
        best_threshold=float(best_payload["threshold"]),
        best_cjk_quality_score=float(best_payload["cjk_quality_score"]),
        best_cjk_visual_score=float(best_payload["groups"]["cjk"]["visual_score"]),
        best_adapted_cjk_f1=float(best_payload["adapted_mask_vs_target_ge2"]["cjk"]["foreground_f1"]),
        best_adapted_cjk_foreground_ratio=float(best_payload["foreground_ratio"]["cjk"]["adapted"]),
        target_cjk_foreground_ratio=float(best_payload["foreground_ratio"]["cjk"]["target_ge2"]),
    )
