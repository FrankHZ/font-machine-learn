from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
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
from font_machine_learn.external_eval import summarize_binary, summarize_visual
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.patch_classifier import model_n_iter, patch_features
from font_machine_learn.paths import (
    SONG13_LAYER_MLP_CONTACT,
    SONG13_LAYER_MLP_ERROR_CONTACT,
    SONG13_LAYER_MLP_METADATA,
    STAGE26_SONG13_LAYER_MLP,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA, Mask, ge2_mask_to_image, make_adapter_contact_sheet
from font_machine_learn.song13_calibrated import mask_foreground_ratio, mean, positive_probabilities
from font_machine_learn.song13_source_locked import levels_to_mask, source_deleted_ratio, source_level_ratios
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class Song13LayerMlpExport:
    target_metadata: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    core_edge_train_pixel_count: int
    shadow_train_pixel_count: int
    best_candidate: str
    best_cjk_quality_score: float
    best_cjk_visual_score: float
    best_source_deleted_ratio: float


class ConstantBinaryModel:
    def __init__(self, value: int) -> None:
        self.value = int(value)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return np.full((len(x_values),), self.value, dtype=np.int64)

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        probability = float(self.value)
        return np.tile(np.asarray([[1.0 - probability, probability]], dtype=np.float32), (len(x_values), 1))


@dataclass(frozen=True)
class EvalGlyph:
    index: int
    codes: list[int]
    chars: list[str]
    source_png: str
    target_png: str
    char_class: str
    group: str
    source_mask: Mask
    target_levels: list[list[int]]
    target_mask: Mask


@dataclass(frozen=True)
class SourceEvaluation:
    name: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_candidate: str
    best_cjk_quality_score: float
    best_cjk_visual_score: float
    best_source_deleted_ratio: float


def local_features(mask: Mask, x: int, y: int, radius: int) -> list[float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    x_scale = max(1, width - 1)
    y_scale = max(1, height - 1)
    neighbors = 0
    right = 0
    down = 0
    diag = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sx = x + dx
            sy = y + dy
            value = 0 <= sx < width and 0 <= sy < height and mask[sy][sx]
            if value:
                neighbors += 1
                if dx == 1 and dy == 0:
                    right = 1
                elif dx == 0 and dy == 1:
                    down = 1
                elif dx == 1 and dy == 1:
                    diag = 1
    return [
        *patch_features(mask, x, y, radius),
        x / x_scale,
        y / y_scale,
        min(x, width - 1 - x) / x_scale,
        min(y, height - 1 - y) / y_scale,
        neighbors / 8.0,
        float(right),
        float(down),
        float(diag),
        float(not right),
        float(not down),
        float(not diag),
    ]


def build_target_ge2_training_rows(
    target_glyphs: list[dict],
    *,
    patch_radius: int,
    max_train_glyphs: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    core_rows: list[list[float]] = []
    core_labels: list[int] = []
    shadow_rows: list[list[float]] = []
    shadow_labels: list[int] = []
    cjk_seen = 0
    for glyph in target_glyphs:
        if classify_glyph(list(glyph["chars"])) != "cjk":
            continue
        if max_train_glyphs is not None and cjk_seen >= max_train_glyphs:
            break
        cjk_seen += 1
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        source_mask = levels_to_threshold_mask(target_levels, "ge2")
        for y, row in enumerate(source_mask):
            for x, is_source in enumerate(row):
                features = local_features(source_mask, x, y, patch_radius)
                if is_source:
                    core_rows.append(features)
                    core_labels.append(1 if target_levels[y][x] == 3 else 0)
                else:
                    shadow_rows.append(features)
                    shadow_labels.append(1 if target_levels[y][x] == 1 else 0)
    return (
        np.asarray(core_rows, dtype=np.float32),
        np.asarray(core_labels, dtype=np.int64),
        np.asarray(shadow_rows, dtype=np.float32),
        np.asarray(shadow_labels, dtype=np.int64),
        cjk_seen,
    )


def balanced_subset(x_train: np.ndarray, y_train: np.ndarray, *, negative_ratio: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y_train == 1)
    negative = np.flatnonzero(y_train == 0)
    if len(positive) == 0 or len(negative) == 0:
        return x_train, y_train
    rng = np.random.default_rng(random_seed)
    negative_count = min(len(negative), len(positive) * negative_ratio)
    selected_negative = rng.choice(negative, size=negative_count, replace=False)
    selected = np.concatenate([positive, selected_negative])
    rng.shuffle(selected)
    return x_train[selected], y_train[selected]


def train_binary_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    prefix: str,
    hidden_units: int,
    max_iter: int,
    random_seed: int,
    negative_ratio: int,
) -> dict[str, object]:
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        value = int(unique_labels[0]) if len(unique_labels) else 0
        return {f"{prefix}_constant": ConstantBinaryModel(value)}

    models: dict[str, object] = {}
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=max(200, max_iter * 4),
        solver="lbfgs",
        random_state=random_seed,
    )
    logistic.fit(x_train, y_train)
    models[f"{prefix}_logistic_balanced"] = logistic

    x_balanced, y_balanced = balanced_subset(
        x_train,
        y_train,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
    )
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
    models[f"{prefix}_patch_mlp"] = mlp
    return models


def probability_grid(model: object, source_mask: Mask, *, patch_radius: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    positions: list[tuple[int, int]] = []
    rows: list[list[float]] = []
    for y, row in enumerate(source_mask):
        for x, _value in enumerate(row):
            positions.append((x, y))
            rows.append(local_features(source_mask, x, y, patch_radius))
    if not rows:
        return positions, np.asarray([], dtype=np.float32)
    return positions, positive_probabilities(model, np.asarray(rows, dtype=np.float32))


def predict_source_locked_levels(
    core_model: object,
    shadow_model: object,
    source_mask: Mask,
    *,
    patch_radius: int,
    core_threshold: float,
    shadow_threshold: float,
) -> list[list[int]]:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]
    positions, core_probabilities = probability_grid(core_model, source_mask, patch_radius=patch_radius)
    _shadow_positions, shadow_probabilities = probability_grid(shadow_model, source_mask, patch_radius=patch_radius)
    for (x, y), core_probability, shadow_probability in zip(positions, core_probabilities, shadow_probabilities):
        if source_mask[y][x]:
            levels[y][x] = 3 if float(core_probability) >= core_threshold else 2
        elif float(shadow_probability) >= shadow_threshold:
            levels[y][x] = 1
    return levels


def load_eval_glyphs(source_glyphs: list[dict]) -> list[EvalGlyph]:
    records: list[EvalGlyph] = []
    for glyph in source_glyphs:
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        records.append(
            EvalGlyph(
                index=int(glyph["index"]),
                codes=list(glyph["codes"]),
                chars=chars,
                source_png=str(glyph["source_png"]),
                target_png=str(glyph["target_png"]),
                char_class=char_class,
                group=group_name(char_class),
                source_mask=image_to_mask(glyph["source_png"]),
                target_levels=target_levels,
                target_mask=levels_to_threshold_mask(target_levels, "visible"),
            )
        )
    return records


def export_target_quantized_source_metadata(
    target_metadata: Path,
    *,
    out_dir: Path,
    mode: str,
) -> Path:
    if mode not in {"ge2", "eq3"}:
        raise ValueError(f"unsupported target quantized mode: {mode}")
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    mask_dir = out_dir / f"target_{mode}_source"
    mask_dir.mkdir(parents=True, exist_ok=True)
    metadata_json = out_dir / f"target_{mode}_source_metadata.json"
    glyphs: list[dict] = []
    for glyph in target["glyphs"]:
        index = int(glyph["index"])
        target_png = glyph["png"]
        target_levels = image_to_target_levels(Image.open(target_png).convert("RGBA"))
        mask = levels_to_threshold_mask(target_levels, mode)
        source_png = mask_dir / f"glyph_{index:04d}.png"
        ge2_mask_to_image(mask).save(source_png)
        glyphs.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": list(glyph["chars"]),
                "source_png": str(source_png),
                "target_png": str(target_png),
            }
        )
    payload = {
        "source_name": f"target_{mode}",
        "source_kind": "target_quantized",
        "target_quantized_mode": mode,
        "cell_width": int(target["cell_width"]),
        "cell_height": int(target["cell_height"]),
        "glyph_count": len(glyphs),
        "glyphs": glyphs,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_json


def predict_source_locked_levels_from_probabilities(
    source_mask: Mask,
    core_probabilities: np.ndarray,
    shadow_probabilities: np.ndarray,
    *,
    core_threshold: float,
    shadow_threshold: float,
) -> list[list[int]]:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]
    offset = 0
    for y in range(height):
        for x in range(width):
            core_probability = float(core_probabilities[offset])
            shadow_probability = float(shadow_probabilities[offset])
            offset += 1
            if source_mask[y][x]:
                levels[y][x] = 3 if core_probability >= core_threshold else 2
            elif shadow_probability >= shadow_threshold:
                levels[y][x] = 1
    return levels


def build_probability_cache(
    source_records: list[EvalGlyph],
    models: dict[str, object],
    *,
    patch_radius: int,
) -> dict[str, dict[int, np.ndarray]]:
    cache: dict[str, dict[int, np.ndarray]] = {name: {} for name in models}
    for glyph in source_records:
        rows: list[list[float]] = []
        for y, row in enumerate(glyph.source_mask):
            for x, _value in enumerate(row):
                rows.append(local_features(glyph.source_mask, x, y, patch_radius))
        feature_matrix = np.asarray(rows, dtype=np.float32)
        for model_name, model in models.items():
            cache[model_name][glyph.index] = positive_probabilities(model, feature_matrix)
    return cache


def summarize_model(model: object) -> dict:
    return {
        "kind": type(model).__name__,
        "n_iter": model_n_iter(model),
    }


def quality_score(candidate: dict) -> float:
    cjk = candidate["groups"]["cjk"]
    contract = candidate["source_contract"]["cjk"]
    ratios = candidate["foreground_ratio"]["cjk"]
    overfill = max(0.0, float(ratios["predicted"]) - float(ratios["target_visible"]) - 0.04)
    return (
        0.42 * float(cjk["visual_score"])
        + 0.22 * float(cjk["ink_f1"])
        + 0.22 * float(cjk["shadow_f1"])
        + 0.14 * float(cjk["foreground_iou"])
        - 0.75 * float(contract["source_deleted_ratio"])
        - 0.30 * overfill
    )


def evaluate_candidate(
    candidate_name: str,
    core_model_name: str,
    shadow_model_name: str,
    core_model: object,
    shadow_model: object,
    source_glyphs: list[EvalGlyph],
    *,
    out_dir: Path | None,
    patch_radius: int,
    core_threshold: float,
    shadow_threshold: float,
    include_glyphs: bool = False,
    core_probability_cache: dict[str, dict[int, np.ndarray]] | None = None,
    shadow_probability_cache: dict[str, dict[int, np.ndarray]] | None = None,
) -> dict:
    source_dir = out_dir / candidate_name / "source_ge2" if out_dir is not None else None
    pred_dir = out_dir / candidate_name / "predicted_2bpp" if out_dir is not None else None
    if source_dir is not None:
        source_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    binary_by_group = {"all": [], "cjk": [], "non_cjk": []}
    deleted_by_group = {"all": [], "cjk": [], "non_cjk": []}
    level2_by_group = {"all": [], "cjk": [], "non_cjk": []}
    level3_by_group = {"all": [], "cjk": [], "non_cjk": []}
    ratios = {
        "all": {"source": [], "predicted": [], "target": []},
        "cjk": {"source": [], "predicted": [], "target": []},
        "non_cjk": {"source": [], "predicted": [], "target": []},
    }
    glyph_payloads: list[dict] = []

    for glyph in source_glyphs:
        index = glyph.index
        source_mask = glyph.source_mask
        if core_probability_cache is not None and shadow_probability_cache is not None:
            predicted_levels = predict_source_locked_levels_from_probabilities(
                source_mask,
                core_probability_cache[core_model_name][index],
                shadow_probability_cache[shadow_model_name][index],
                core_threshold=core_threshold,
                shadow_threshold=shadow_threshold,
            )
        else:
            predicted_levels = predict_source_locked_levels(
                core_model,
                shadow_model,
                source_mask,
                patch_radius=patch_radius,
                core_threshold=core_threshold,
                shadow_threshold=shadow_threshold,
            )
        predicted_mask = levels_to_mask(predicted_levels)

        source_png = source_dir / f"glyph_{index:04d}.png" if source_dir is not None else None
        predicted_png = pred_dir / f"glyph_{index:04d}.png" if pred_dir is not None else None
        if source_png is not None:
            ge2_mask_to_image(source_mask).save(source_png)
        if predicted_png is not None:
            levels_to_image(predicted_levels).save(predicted_png)

        binary = compare_masks(predicted_mask, glyph.target_mask)
        visual = compare_visual(predicted_levels, glyph.target_levels)
        deleted = source_deleted_ratio(source_mask, predicted_levels)
        level_ratios = source_level_ratios(source_mask, predicted_levels)
        for key in ("all", glyph.group):
            visual_by_group[key].append(visual)
            binary_by_group[key].append(binary)
            deleted_by_group[key].append(deleted)
            level2_by_group[key].append(level_ratios["level2"])
            level3_by_group[key].append(level_ratios["level3"])
            ratios[key]["source"].append(mask_foreground_ratio(source_mask))
            ratios[key]["predicted"].append(mask_foreground_ratio(predicted_mask))
            ratios[key]["target"].append(mask_foreground_ratio(glyph.target_mask))

        if include_glyphs:
            glyph_payloads.append(
                {
                    "index": index,
                    "codes": glyph.codes,
                    "chars": glyph.chars,
                    "original_source_png": glyph.source_png,
                    "source_png": str(source_png) if source_png is not None else glyph.source_png,
                    "predicted_png": str(predicted_png) if predicted_png is not None else "",
                    "target_png": glyph.target_png,
                    "char_class": glyph.char_class,
                    "source_deleted_ratio": deleted,
                    "source_level_ratio": level_ratios,
                    "binary": asdict(binary),
                    "visual": visual.to_dict(),
                    "foreground_ratio": {
                        "source": mask_foreground_ratio(source_mask),
                        "predicted": mask_foreground_ratio(predicted_mask),
                        "target_visible": mask_foreground_ratio(glyph.target_mask),
                    },
                }
            )

    payload = {
        "core_model": core_model_name,
        "shadow_model": shadow_model_name,
        "core_threshold": core_threshold,
        "shadow_threshold": shadow_threshold,
        "source_dir": str(source_dir) if source_dir is not None else "",
        "predicted_dir": str(pred_dir) if pred_dir is not None else "",
        "groups": {key: summarize_visual(visual_by_group[key]) for key in ("all", "cjk", "non_cjk")},
        "binary": {key: summarize_binary(binary_by_group[key]) for key in ("all", "cjk", "non_cjk")},
        "foreground_ratio": {
            key: {
                "source": mean(ratios[key]["source"]),
                "predicted": mean(ratios[key]["predicted"]),
                "target_visible": mean(ratios[key]["target"]),
            }
            for key in ("all", "cjk", "non_cjk")
        },
        "source_contract": {
            key: {
                "source_deleted_ratio": mean(deleted_by_group[key]),
                "source_level2_ratio": mean(level2_by_group[key]),
                "source_level3_ratio": mean(level3_by_group[key]),
            }
            for key in ("all", "cjk", "non_cjk")
        },
        "glyphs": glyph_payloads,
    }
    payload["cjk_quality_score"] = quality_score(payload)
    return payload


def evaluate_candidate_specs(
    specs: list[tuple[str, str, str, object, object, float, float]],
    source_records: list[EvalGlyph],
    *,
    patch_radius: int,
    core_probability_cache: dict[str, dict[int, np.ndarray]],
    shadow_probability_cache: dict[str, dict[int, np.ndarray]],
    jobs: int,
) -> dict[str, dict]:
    def run_spec(spec: tuple[str, str, str, object, object, float, float]) -> tuple[str, dict]:
        candidate_name, core_model_name, shadow_model_name, core_model, shadow_model, core_threshold, shadow_threshold = spec
        return candidate_name, evaluate_candidate(
            candidate_name,
            core_model_name,
            shadow_model_name,
            core_model,
            shadow_model,
            source_records,
            out_dir=None,
            patch_radius=patch_radius,
            core_threshold=core_threshold,
            shadow_threshold=shadow_threshold,
            include_glyphs=False,
            core_probability_cache=core_probability_cache,
            shadow_probability_cache=shadow_probability_cache,
        )

    if jobs <= 1 or len(specs) <= 1:
        return dict(run_spec(spec) for spec in specs)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        return dict(executor.map(run_spec, specs))


def evaluate_layer_source(
    *,
    source_name: str,
    source_metadata: Path,
    source_payload: dict,
    core_models: dict[str, object],
    shadow_models: dict[str, object],
    specs: list[tuple[str, str, str, object, object, float, float]],
    candidate_specs: dict[str, tuple[str, str, object, object, float, float]],
    patch_radius: int,
    search_limit: int | None,
    candidate_jobs: int,
    out_dir: Path,
    metadata_json: Path,
    contact_sheet: Path,
    error_contact_sheet: Path,
    worst_count: int,
    scale: int,
    columns: int,
    pad: int,
) -> tuple[SourceEvaluation, dict]:
    source_records = load_eval_glyphs(list(source_payload["glyphs"]))
    cell_width = int(source_payload["cell_width"])
    cell_height = int(source_payload["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    search_records = source_records[:search_limit] if search_limit is not None else source_records
    core_probability_cache = build_probability_cache(source_records, core_models, patch_radius=patch_radius)
    shadow_probability_cache = build_probability_cache(source_records, shadow_models, patch_radius=patch_radius)
    candidates = evaluate_candidate_specs(
        specs,
        search_records,
        patch_radius=patch_radius,
        core_probability_cache=core_probability_cache,
        shadow_probability_cache=shadow_probability_cache,
        jobs=max(1, candidate_jobs),
    )
    best_candidate = max(candidates, key=lambda name: float(candidates[name]["cjk_quality_score"]))
    best_core_name, best_shadow_name, best_core_model, best_shadow_model, best_core_threshold, best_shadow_threshold = (
        candidate_specs[best_candidate]
    )
    best_payload = evaluate_candidate(
        best_candidate,
        best_core_name,
        best_shadow_name,
        best_core_model,
        best_shadow_model,
        source_records,
        out_dir=out_dir,
        patch_radius=patch_radius,
        core_threshold=best_core_threshold,
        shadow_threshold=best_shadow_threshold,
        include_glyphs=True,
        core_probability_cache=core_probability_cache,
        shadow_probability_cache=shadow_probability_cache,
    )
    candidates[best_candidate] = best_payload
    cjk_records = [glyph for glyph in best_payload["glyphs"] if glyph["char_class"] == "cjk"]
    make_adapter_contact_sheet(
        cjk_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst_records = sorted(cjk_records, key=lambda record: float(record["visual"]["visual_score"]))[:worst_count]
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
        "source_name": source_name,
        "source_metadata": str(source_metadata),
        "glyph_count": len(source_records),
        "cjk_glyph_count": len(cjk_records),
        "best_candidate": best_candidate,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_candidate": best_candidate,
        "contact_sheet_order": ["original_source", "source_ge2", "predicted_2bpp", "target_2bpp"],
        "candidates": candidates,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = SourceEvaluation(
        name=source_name,
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(source_records),
        cjk_glyph_count=len(cjk_records),
        best_candidate=best_candidate,
        best_cjk_quality_score=float(best_payload["cjk_quality_score"]),
        best_cjk_visual_score=float(best_payload["groups"]["cjk"]["visual_score"]),
        best_source_deleted_ratio=float(best_payload["source_contract"]["cjk"]["source_deleted_ratio"]),
    )
    return result, payload


def export_song13_layer_mlp(
    target_metadata: Path = TARGET_METADATA,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    *,
    out_dir: Path = STAGE26_SONG13_LAYER_MLP,
    metadata_json: Path = SONG13_LAYER_MLP_METADATA,
    contact_sheet: Path = SONG13_LAYER_MLP_CONTACT,
    error_contact_sheet: Path = SONG13_LAYER_MLP_ERROR_CONTACT,
    core_thresholds: list[float] | None = None,
    shadow_thresholds: list[float] | None = None,
    patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    search_limit: int | None = 512,
    jobs: int = 1,
    extra_eval_sources: dict[str, Path] | None = None,
    eval_source_jobs: int = 1,
    core_hidden_units: int = 64,
    shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 26,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13LayerMlpExport:
    if core_thresholds is None:
        core_thresholds = [0.45, 0.50, 0.55, 0.65]
    if shadow_thresholds is None:
        shadow_thresholds = [0.45, 0.55, 0.65, 0.75]
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
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    x_core, y_core, x_shadow, y_shadow, train_cjk = build_target_ge2_training_rows(
        target_glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    core_models = train_binary_models(
        x_core,
        y_core,
        prefix="core",
        hidden_units=core_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
        negative_ratio=3,
    )
    shadow_models = train_binary_models(
        x_shadow,
        y_shadow,
        prefix="shadow",
        hidden_units=shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
        negative_ratio=3,
    )

    specs: list[tuple[str, str, str, object, object, float, float]] = []
    candidate_specs: dict[str, tuple[str, str, object, object, float, float]] = {}
    for core_model_name, core_model in core_models.items():
        for shadow_model_name, shadow_model in shadow_models.items():
            for core_threshold in core_thresholds:
                for shadow_threshold in shadow_thresholds:
                    candidate_name = (
                        f"{core_model_name}_{shadow_model_name}"
                        f"_c{int(round(core_threshold * 100)):03d}"
                        f"_s{int(round(shadow_threshold * 100)):03d}"
                    )
                    candidate_specs[candidate_name] = (
                        core_model_name,
                        shadow_model_name,
                        core_model,
                        shadow_model,
                        float(core_threshold),
                        float(shadow_threshold),
                    )
                    specs.append(
                        (
                            candidate_name,
                            core_model_name,
                            shadow_model_name,
                            core_model,
                            shadow_model,
                            float(core_threshold),
                            float(shadow_threshold),
                        )
                    )

    primary_source_name = str(source.get("source_name", "song13"))
    source_configs: list[tuple[str, Path, dict, Path, Path, Path, Path]] = [
        (primary_source_name, source_metadata, source, out_dir, metadata_json, contact_sheet, error_contact_sheet)
    ]
    if extra_eval_sources:
        for name, extra_metadata in extra_eval_sources.items():
            extra_source = json.loads(extra_metadata.read_text(encoding="utf-8"))
            source_root = out_dir / "eval_sources" / name
            source_configs.append(
                (
                    name,
                    extra_metadata,
                    extra_source,
                    source_root,
                    source_root / f"{name}_metadata.json",
                    source_root / f"{name}_contact.png",
                    source_root / f"{name}_errors.png",
                )
            )

    def run_source(config: tuple[str, Path, dict, Path, Path, Path, Path]) -> tuple[str, SourceEvaluation, dict]:
        name, metadata_path, source_payload, source_out, source_json, source_contact, source_errors = config
        result, source_result_payload = evaluate_layer_source(
            source_name=name,
            source_metadata=metadata_path,
            source_payload=source_payload,
            core_models=core_models,
            shadow_models=shadow_models,
            specs=specs,
            candidate_specs=candidate_specs,
            patch_radius=patch_radius,
            search_limit=search_limit,
            candidate_jobs=max(1, jobs),
            out_dir=source_out,
            metadata_json=source_json,
            contact_sheet=source_contact,
            error_contact_sheet=source_errors,
            worst_count=worst_count,
            scale=scale,
            columns=columns,
            pad=pad,
        )
        return name, result, source_result_payload

    if eval_source_jobs > 1 and len(source_configs) > 1:
        with ThreadPoolExecutor(max_workers=eval_source_jobs) as executor:
            source_results = list(executor.map(run_source, source_configs))
    else:
        source_results = [run_source(config) for config in source_configs]

    source_result_by_name = {name: result for name, result, _payload in source_results}
    source_payload_by_name = {name: payload for name, _result, payload in source_results}
    primary_result = source_result_by_name[primary_source_name]
    primary_payload = source_payload_by_name[primary_source_name]

    payload = {
        "target_metadata": str(target_metadata),
        "source_metadata": str(source_metadata),
        "glyph_count": primary_result.glyph_count,
        "cjk_glyph_count": primary_result.cjk_glyph_count,
        "task": "source-locked Song13 1bpp mask -> learned NFTR-style 2bpp layer assignment",
        "training_contract": "Train core/edge and shadow heads on target-derived ge2 masks; apply them to Song13 without changing source shape.",
        "source_contract": "Song13 source pixels are never deleted; source pixels become level 2 or 3, and outside-source pixels can only become level 1 shadow.",
        "target_training_source": "target level >= 2 is used only to learn layer semantics, not to adapt Song13 glyph shape",
        "patch_radius": patch_radius,
        "patch_size": patch_radius * 2 + 1,
        "core_thresholds": core_thresholds,
        "shadow_thresholds": shadow_thresholds,
        "search_limit": search_limit,
        "jobs": jobs,
        "eval_source_jobs": eval_source_jobs,
        "train_cjk_glyph_count": train_cjk,
        "train_pixel_count": {
            "core_edge": int(len(y_core)),
            "shadow": int(len(y_shadow)),
        },
        "label_counts": {
            "core_edge": {str(value): int(np.count_nonzero(y_core == value)) for value in (0, 1)},
            "shadow": {str(value): int(np.count_nonzero(y_shadow == value)) for value in (0, 1)},
        },
        "models": {
            "core_edge": {name: summarize_model(model) for name, model in core_models.items()},
            "shadow": {name: summarize_model(model) for name, model in shadow_models.items()},
        },
        "selection": {
            "primary": "cjk_quality_score",
            "formula": "0.42*visual + 0.22*ink + 0.22*shadow + 0.14*foreground_iou - source_deletion_penalty - overfill_penalty",
            "reason": "Stage26 should improve layer assignment while keeping Stage25's source-preservation contract.",
        },
        "best_candidate": primary_result.best_candidate,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_candidate": primary_result.best_candidate,
        "contact_sheet_order": ["original_source", "source_ge2", "predicted_2bpp", "target_2bpp"],
        "primary_source": asdict(primary_result),
        "eval_sources": {
            name: asdict(result)
            for name, result in source_result_by_name.items()
            if name != primary_source_name
        },
        "candidates": primary_payload["candidates"],
        "interpretation_notes": [
            "This stage deliberately does not predict a new ge2 mask for Song13.",
            "Low target-overlap scores can still be acceptable because Song13 and the NFTR target have different glyph shapes.",
            "Compare against Stage25 to decide whether learned 2/3 and 0/1 placement looks less mechanical by eye.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return Song13LayerMlpExport(
        target_metadata=str(target_metadata),
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=primary_result.glyph_count,
        cjk_glyph_count=primary_result.cjk_glyph_count,
        core_edge_train_pixel_count=int(len(y_core)),
        shadow_train_pixel_count=int(len(y_shadow)),
        best_candidate=primary_result.best_candidate,
        best_cjk_quality_score=primary_result.best_cjk_quality_score,
        best_cjk_visual_score=primary_result.best_cjk_visual_score,
        best_source_deleted_ratio=primary_result.best_source_deleted_ratio,
    )
