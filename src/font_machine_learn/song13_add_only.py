from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import group_name
from font_machine_learn.external_eval import summarize_binary, summarize_visual, train_two_head_models
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    SONG13_ADD_ONLY_CONTACT,
    SONG13_ADD_ONLY_ERROR_CONTACT,
    SONG13_ADD_ONLY_METADATA,
    STAGE24_SONG13_ADD_ONLY,
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
    ge2_mask_to_image,
    make_adapter_contact_sheet,
)
from font_machine_learn.song13_calibrated import (
    adapter_feature_matrix,
    mask_foreground_ratio,
    mean,
    positive_probabilities,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class Song13AddOnlyExport:
    target_metadata: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    train_pixel_count: int
    best_candidate: str
    best_model: str
    best_threshold: float
    best_cjk_quality_score: float
    best_cjk_visual_score: float
    best_adapted_cjk_f1: float
    best_source_deleted_ratio: float


class ConstantAddModel:
    def __init__(self, value: int) -> None:
        self.value = int(value)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return np.full((len(x_values),), self.value, dtype=np.int64)

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        probability = float(self.value)
        return np.tile(np.asarray([[1.0 - probability, probability]], dtype=np.float32), (len(x_values), 1))


def build_add_training_rows(
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
            for x, value in enumerate(row):
                if value:
                    continue
                rows.append(adapter_features(source_mask, x, y, patch_radius))
                labels.append(1 if target_ge2[y][x] else 0)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), cjk_seen


def balanced_add_subset(x_train: np.ndarray, y_train: np.ndarray, *, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y_train == 1)
    negative = np.flatnonzero(y_train == 0)
    if len(positive) == 0 or len(negative) == 0:
        return x_train, y_train
    rng = np.random.default_rng(random_seed)
    negative_count = min(len(negative), len(positive) * 3)
    selected_negative = rng.choice(negative, size=negative_count, replace=False)
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    return x_train[selected], y_train[selected]


def train_add_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> dict[str, object]:
    models: dict[str, object] = {}
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        value = int(unique_labels[0]) if len(unique_labels) else 0
        models["add_constant"] = ConstantAddModel(value)
        return models
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=max(200, max_iter * 4),
        solver="lbfgs",
        random_state=random_seed,
    )
    logistic.fit(x_train, y_train)
    models["add_logistic_balanced"] = logistic

    x_balanced, y_balanced = balanced_add_subset(x_train, y_train, random_seed=random_seed)
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
    models["add_patch_mlp"] = mlp
    return models


def add_only_mask(model: object, source_mask: Mask, *, patch_radius: int, threshold: float) -> Mask:
    rows, positions = adapter_feature_matrix(source_mask, patch_radius=patch_radius)
    probabilities = positive_probabilities(model, rows)
    adapted = [row[:] for row in source_mask]
    for (x, y), probability in zip(positions, probabilities):
        if source_mask[y][x]:
            continue
        if float(probability) >= threshold:
            adapted[y][x] = True
    return adapted


def source_deleted_ratio(source_mask: Mask, adapted_mask: Mask) -> float:
    source_count = sum(1 for row in source_mask for value in row if value)
    if source_count == 0:
        return 0.0
    deleted = sum(
        1
        for y, row in enumerate(source_mask)
        for x, value in enumerate(row)
        if value and not adapted_mask[y][x]
    )
    return deleted / source_count


def quality_score(candidate: dict) -> float:
    groups = candidate["groups"]["cjk"]
    adapted = candidate["adapted_mask_vs_target_ge2"]["cjk"]
    ratios = candidate["foreground_ratio"]["cjk"]
    ratio_penalty = max(0.0, float(ratios["adapted"]) - float(ratios["target_ge2"]))
    return (
        0.45 * float(groups["visual_score"])
        + 0.25 * float(adapted["foreground_f1"])
        + 0.15 * float(adapted["foreground_precision"])
        + 0.15 * float(groups["ink_f1"])
        - 0.80 * ratio_penalty
    )


def evaluate_add_candidate(
    candidate_name: str,
    model_name: str,
    model: object,
    threshold: float,
    source_glyphs: list[dict],
    *,
    out_dir: Path,
    core_model: object,
    shadow_model: object,
    add_patch_radius: int,
    style_patch_radius: int,
) -> dict:
    adapted_dir = out_dir / candidate_name / "adapted_ge2"
    pred_dir = out_dir / candidate_name / "predicted_2bpp"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    source_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    adapted_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    deleted_by_group = {"all": [], "cjk": [], "non_cjk": []}
    ratios = {
        "all": {"source": [], "adapted": [], "target_ge2": []},
        "cjk": {"source": [], "adapted": [], "target_ge2": []},
        "non_cjk": {"source": [], "adapted": [], "target_ge2": []},
    }
    shadow_matrix = {str(actual): {str(value): 0 for value in (0, 1)} for actual in (0, 1)}
    glyph_payloads: list[dict] = []

    for glyph in source_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = group_name(char_class)
        source_mask = image_to_mask(glyph["source_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_ge2 = levels_to_threshold_mask(target_levels, "ge2")
        adapted_mask = add_only_mask(model, source_mask, patch_radius=add_patch_radius, threshold=threshold)

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
        deleted = source_deleted_ratio(source_mask, adapted_mask)
        for key in ("all", group):
            visual_by_group[key].append(visual)
            source_mask_by_group[key].append(source_mask_metrics)
            adapted_mask_by_group[key].append(adapted_mask_metrics)
            deleted_by_group[key].append(deleted)
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
                "source_deleted_ratio": deleted,
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
        "model": model_name,
        "threshold": threshold,
        "adapted_dir": str(adapted_dir),
        "predicted_dir": str(pred_dir),
        "source_deleted_ratio": {
            key: mean(deleted_by_group[key])
            for key in ("all", "cjk", "non_cjk")
        },
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
    return payload


def export_song13_add_only(
    target_metadata: Path = TARGET_METADATA,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    *,
    out_dir: Path = STAGE24_SONG13_ADD_ONLY,
    metadata_json: Path = SONG13_ADD_ONLY_METADATA,
    contact_sheet: Path = SONG13_ADD_ONLY_CONTACT,
    error_contact_sheet: Path = SONG13_ADD_ONLY_ERROR_CONTACT,
    thresholds: list[float] | None = None,
    add_patch_radius: int = 3,
    style_patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    add_hidden_units: int = 48,
    style_core_hidden_units: int = 64,
    style_shadow_hidden_units: int = 64,
    max_iter: int = 60,
    random_seed: int = 24,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13AddOnlyExport:
    if thresholds is None:
        thresholds = [0.65, 0.75]
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
    x_add, y_add, add_train_glyph_count = build_add_training_rows(
        source_glyphs,
        patch_radius=add_patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    models = train_add_models(
        x_add,
        y_add,
        hidden_units=add_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )

    candidates: dict[str, dict] = {}
    for model_name, model in models.items():
        for threshold in thresholds:
            candidate_name = f"{model_name}_t{int(round(threshold * 100)):03d}"
            candidates[candidate_name] = evaluate_add_candidate(
                candidate_name,
                model_name,
                model,
                float(threshold),
                source_glyphs,
                out_dir=out_dir,
                core_model=core_model,
                shadow_model=shadow_model,
                add_patch_radius=add_patch_radius,
                style_patch_radius=style_patch_radius,
            )

    best_candidate = max(candidates, key=lambda name: float(candidates[name]["cjk_quality_score"]))
    best_payload = candidates[best_candidate]
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
        "task": "source-preserving Song13 add-only ge2 adapter before Stage20 style heads",
        "source_contract": "WenQuanYi Bitmap Song 13px, rendered at font-size 15 and left-bottom aligned",
        "source_preservation": "adapted_ge2 always includes every original Song13 source pixel; the model can only add outside-source ge2 pixels",
        "adapted_ge2_visualization": "foreground pixels are saved as NFTR level 2 gray for readability; metrics treat this as a binary ge2 mask",
        "selection": {
            "primary": "cjk_quality_score",
            "formula": "0.45*visual + 0.25*adapted_f1 + 0.15*adapted_precision + 0.15*ink_f1 - 0.80*max(0, adapted_ratio-target_ge2_ratio)",
            "reason": "Stage22/23 deleted Song13 strokes; Stage24 forbids deletion and only tunes additions.",
        },
        "thresholds": thresholds,
        "add_patch_radius": add_patch_radius,
        "add_patch_size": add_patch_radius * 2 + 1,
        "style_patch_radius": style_patch_radius,
        "style_patch_size": style_patch_radius * 2 + 1,
        "train_cjk_glyph_count": {
            "add": add_train_glyph_count,
            **style_train_payload["train_cjk_glyph_count"],
        },
        "train_pixel_count": {
            "add": int(len(y_add)),
            **style_train_payload["train_pixel_count"],
        },
        "label_counts": {
            "add": {str(value): int(np.count_nonzero(y_add == value)) for value in (0, 1)},
            **style_train_payload["label_counts"],
        },
        "best_candidate": best_candidate,
        "best_model": best_payload["model"],
        "best_threshold": best_payload["threshold"],
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_candidate": best_candidate,
        "contact_sheet_order": ["original_source", "adapted_ge2", "predicted_2bpp", "target_2bpp"],
        "candidates": candidates,
        "interpretation_notes": [
            "Stage22/23 adapters were supervised by target ge2 and deleted source strokes, effectively learning target glyph-shape correction.",
            "Stage24 is a diagnostic guardrail: it cannot delete source strokes, so failures show what remains when Song13 shape is treated as the contract.",
            "If Stage24 is more readable but lower scoring, the next model should learn layer assignment from source shape rather than target-shape replacement.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return Song13AddOnlyExport(
        target_metadata=str(target_metadata),
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(source_glyphs),
        cjk_glyph_count=len(best_contact_records),
        train_pixel_count=int(len(y_add)),
        best_candidate=best_candidate,
        best_model=str(best_payload["model"]),
        best_threshold=float(best_payload["threshold"]),
        best_cjk_quality_score=float(best_payload["cjk_quality_score"]),
        best_cjk_visual_score=float(best_payload["groups"]["cjk"]["visual_score"]),
        best_adapted_cjk_f1=float(best_payload["adapted_mask_vs_target_ge2"]["cjk"]["foreground_f1"]),
        best_source_deleted_ratio=float(best_payload["source_deleted_ratio"]["cjk"]),
    )
