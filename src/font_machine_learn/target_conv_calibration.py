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
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.external_eval import summarize_visual
from font_machine_learn.nftr import checkerboard, export_target_dataset
from font_machine_learn.patch_classifier import model_n_iter
from font_machine_learn.paths import TARGET_CONV_CONTACT, TARGET_CONV_ERROR_CONTACT, TARGET_CONV_METADATA, STAGE29_TARGET_CONV, TARGET_METADATA
from font_machine_learn.song13_calibrated import positive_probabilities
from font_machine_learn.song13_layer_mlp import balanced_subset
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


Mask = list[list[bool]]
LevelGrid = list[list[int]]


@dataclass(frozen=True)
class TargetConvExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    best_candidate: str
    best_cjk_visual_score: float


@dataclass(frozen=True)
class EvalGlyph:
    index: int
    codes: list[int]
    chars: list[str]
    char_class: str
    group: str
    target_png: str
    target_levels: LevelGrid
    source_mask: Mask


def _mask_array(mask: Mask) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32)


def _window_sum(padded: np.ndarray, x: int, y: int, radius: int) -> float:
    patch = padded[y : y + radius * 2 + 1, x : x + radius * 2 + 1]
    return float(np.sum(patch))


def conv_features(mask: Mask, x: int, y: int) -> list[float]:
    arr = _mask_array(mask)
    height, width = arr.shape if arr.size else (0, 0)
    padded1 = np.pad(arr, 1)
    padded2 = np.pad(arr, 2)
    px = x + 1
    py = y + 1
    patch = padded1[py - 1 : py + 2, px - 1 : px + 2]
    center = float(patch[1, 1])
    neighbors = float(np.sum(patch) - center)
    right = float(patch[1, 2])
    down = float(patch[2, 1])
    diag = float(patch[2, 2])
    left = float(patch[1, 0])
    up = float(patch[0, 1])
    up_left = float(patch[0, 0])
    up_right = float(patch[0, 2])
    down_left = float(patch[2, 0])
    sobel_x = float((-patch[0, 0] - 2 * patch[1, 0] - patch[2, 0]) + (patch[0, 2] + 2 * patch[1, 2] + patch[2, 2]))
    sobel_y = float((-patch[0, 0] - 2 * patch[0, 1] - patch[0, 2]) + (patch[2, 0] + 2 * patch[2, 1] + patch[2, 2]))
    laplacian = float(4 * center - up - down - left - right)
    sum5 = _window_sum(padded2, x, y, 2)
    x_scale = max(1, width - 1)
    y_scale = max(1, height - 1)
    return [
        center,
        neighbors / 8.0,
        sum5 / 25.0,
        right,
        down,
        diag,
        left,
        up,
        up_left,
        up_right,
        down_left,
        sobel_x / 4.0,
        sobel_y / 4.0,
        laplacian / 4.0,
        x / x_scale,
        y / y_scale,
        min(x, width - 1 - x) / x_scale,
        min(y, height - 1 - y) / y_scale,
    ]


def build_training_rows(
    target_glyphs: list[dict],
    *,
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
                features = conv_features(source_mask, x, y)
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


def load_eval_glyphs(target_glyphs: list[dict]) -> list[EvalGlyph]:
    records: list[EvalGlyph] = []
    for glyph in target_glyphs:
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        target_png = str(glyph["png"])
        target_levels = image_to_target_levels(Image.open(target_png).convert("RGBA"))
        records.append(
            EvalGlyph(
                index=int(glyph["index"]),
                codes=list(glyph["codes"]),
                chars=chars,
                char_class=char_class,
                group="cjk" if char_class == "cjk" else "non_cjk",
                target_png=target_png,
                target_levels=target_levels,
                source_mask=levels_to_threshold_mask(target_levels, "ge2"),
            )
        )
    return records


def train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    prefix: str,
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
    models[f"{prefix}_conv_logistic"] = logistic

    x_balanced, y_balanced = balanced_subset(
        x_train,
        y_train,
        negative_ratio=3,
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
    models[f"{prefix}_conv_mlp"] = mlp
    return models


def probability_grid(model: object, source_mask: Mask) -> np.ndarray:
    rows = [
        conv_features(source_mask, x, y)
        for y, row in enumerate(source_mask)
        for x, _value in enumerate(row)
    ]
    return positive_probabilities(model, np.asarray(rows, dtype=np.float32))


def predict_levels(
    source_mask: Mask,
    core_probabilities: np.ndarray,
    shadow_probabilities: np.ndarray,
    *,
    core_threshold: float,
    shadow_threshold: float,
) -> LevelGrid:
    height = len(source_mask)
    width = len(source_mask[0]) if height else 0
    levels = [[0 for _x in range(width)] for _y in range(height)]
    offset = 0
    for y, row in enumerate(source_mask):
        for x, is_source in enumerate(row):
            if is_source:
                levels[y][x] = 3 if float(core_probabilities[offset]) >= core_threshold else 2
            elif float(shadow_probabilities[offset]) >= shadow_threshold:
                levels[y][x] = 1
            offset += 1
    return levels


def make_contact_sheet(
    rows: list[dict],
    *,
    out_path: Path,
    cell_width: int,
    cell_height: int,
    scale: int,
    columns: int,
    pad: int,
) -> None:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    group_h = tile_h * 3 + pad * 2
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, sheet_rows * group_h + (sheet_rows + 1) * pad),
        max(2, scale * 2),
    )
    for position, row in enumerate(rows):
        x = pad + (position % columns) * (tile_w + pad)
        y = pad + (position // columns) * (group_h + pad)
        for image_index, key in enumerate(("source_png", "predicted_png", "target_png")):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + image_index * (tile_h + pad)))
    sheet.save(out_path)


def evaluate_candidate(
    *,
    candidate_name: str,
    core_model_name: str,
    shadow_model_name: str,
    core_probabilities: dict[int, np.ndarray],
    shadow_probabilities: dict[int, np.ndarray],
    eval_glyphs: list[EvalGlyph],
    out_dir: Path | None,
    core_threshold: float,
    shadow_threshold: float,
) -> dict:
    source_dir = out_dir / "source_ge2" if out_dir is not None else None
    pred_dir = out_dir / "predicted_2bpp" if out_dir is not None else None
    if source_dir is not None:
        source_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    records: list[dict] = []
    for glyph in eval_glyphs:
        index = glyph.index
        predicted_levels = predict_levels(
            glyph.source_mask,
            core_probabilities[index],
            shadow_probabilities[index],
            core_threshold=core_threshold,
            shadow_threshold=shadow_threshold,
        )
        visual = compare_visual(predicted_levels, glyph.target_levels)
        for key in ("all", glyph.group):
            visual_by_group[key].append(visual)

        source_png = source_dir / f"glyph_{index:04d}.png" if source_dir is not None else None
        predicted_png = pred_dir / f"glyph_{index:04d}.png" if pred_dir is not None else None
        if source_png is not None:
            levels_to_image([[2 if value else 0 for value in row] for row in glyph.source_mask]).save(source_png)
        if predicted_png is not None:
            levels_to_image(predicted_levels).save(predicted_png)
        records.append(
            {
                "index": index,
                "codes": glyph.codes,
                "chars": glyph.chars,
                "char_class": glyph.char_class,
                "source_png": str(source_png) if source_png is not None else "",
                "predicted_png": str(predicted_png) if predicted_png is not None else "",
                "target_png": glyph.target_png,
                "visual": visual.to_dict(),
            }
        )
    return {
        "candidate": candidate_name,
        "core_model": core_model_name,
        "shadow_model": shadow_model_name,
        "core_threshold": core_threshold,
        "shadow_threshold": shadow_threshold,
        "groups": {key: summarize_visual(visual_by_group[key]) for key in ("all", "cjk", "non_cjk")},
        "glyphs": records,
    }


def export_target_conv_calibration(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE29_TARGET_CONV,
    metadata_json: Path = TARGET_CONV_METADATA,
    contact_sheet: Path = TARGET_CONV_CONTACT,
    error_contact_sheet: Path = TARGET_CONV_ERROR_CONTACT,
    core_thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60),
    shadow_thresholds: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65),
    max_train_glyphs: int | None = None,
    hidden_units: int = 32,
    max_iter: int = 80,
    random_seed: int = 29,
    contact_count: int = 160,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> TargetConvExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    target_glyphs = list(target["glyphs"])
    eval_glyphs = load_eval_glyphs(target_glyphs)
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    x_core, y_core, x_shadow, y_shadow, train_cjk = build_training_rows(
        target_glyphs,
        max_train_glyphs=max_train_glyphs,
    )
    core_models = train_models(
        x_core,
        y_core,
        prefix="core",
        hidden_units=hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )
    shadow_models = train_models(
        x_shadow,
        y_shadow,
        prefix="shadow",
        hidden_units=hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )

    core_probabilities = {
        name: {} for name in core_models
    }
    shadow_probabilities = {
        name: {} for name in shadow_models
    }
    for glyph in eval_glyphs:
        index = glyph.index
        for name, model in core_models.items():
            core_probabilities[name][index] = probability_grid(model, glyph.source_mask)
        for name, model in shadow_models.items():
            shadow_probabilities[name][index] = probability_grid(model, glyph.source_mask)

    search_payloads: dict[str, dict] = {}
    for core_name in core_models:
        for shadow_name in shadow_models:
            for core_threshold in core_thresholds:
                for shadow_threshold in shadow_thresholds:
                    candidate_name = (
                        f"{core_name}_{shadow_name}"
                        f"_c{int(round(core_threshold * 100)):03d}"
                        f"_s{int(round(shadow_threshold * 100)):03d}"
                    )
                    payload = evaluate_candidate(
                        candidate_name=candidate_name,
                        core_model_name=core_name,
                        shadow_model_name=shadow_name,
                        core_probabilities=core_probabilities[core_name],
                        shadow_probabilities=shadow_probabilities[shadow_name],
                        eval_glyphs=eval_glyphs,
                        out_dir=None,
                        core_threshold=core_threshold,
                        shadow_threshold=shadow_threshold,
                    )
                    search_payloads[candidate_name] = payload

    best_candidate = max(search_payloads, key=lambda name: float(search_payloads[name]["groups"]["cjk"]["visual_score"]))
    best_search = search_payloads[best_candidate]
    best_dir = out_dir / best_candidate
    best_payload = evaluate_candidate(
        candidate_name=best_candidate,
        core_model_name=str(best_search["core_model"]),
        shadow_model_name=str(best_search["shadow_model"]),
        core_probabilities=core_probabilities[str(best_search["core_model"])],
        shadow_probabilities=shadow_probabilities[str(best_search["shadow_model"])],
        eval_glyphs=eval_glyphs,
        out_dir=best_dir,
        core_threshold=float(best_search["core_threshold"]),
        shadow_threshold=float(best_search["shadow_threshold"]),
    )
    search_payloads[best_candidate] = best_payload
    cjk_records = [record for record in best_payload["glyphs"] if record["char_class"] == "cjk"]
    make_contact_sheet(
        cjk_records[:contact_count],
        out_path=contact_sheet,
        cell_width=cell_width,
        cell_height=cell_height,
        scale=scale,
        columns=columns,
        pad=pad,
    )
    worst_records = sorted(cjk_records, key=lambda record: float(record["visual"]["visual_score"]))[:worst_count]
    make_contact_sheet(
        worst_records,
        out_path=error_contact_sheet,
        cell_width=cell_width,
        cell_height=cell_height,
        scale=scale,
        columns=columns,
        pad=pad,
    )

    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(target_glyphs),
        "cjk_glyph_count": len(cjk_records),
        "task": "small convolution-feature style recovery from target >=2 source masks",
        "source_mask": "target level >= 2",
        "feature_set": "small 3x3/5x5 convolution responses plus normalized coordinates",
        "train_cjk_glyph_count": train_cjk,
        "train_pixel_count": {"core_edge": int(len(y_core)), "shadow": int(len(y_shadow))},
        "label_counts": {
            "core_edge": {str(value): int(np.count_nonzero(y_core == value)) for value in (0, 1)},
            "shadow": {str(value): int(np.count_nonzero(y_shadow == value)) for value in (0, 1)},
        },
        "models": {
            "core_edge": {name: {"kind": type(model).__name__, "n_iter": model_n_iter(model)} for name, model in core_models.items()},
            "shadow": {name: {"kind": type(model).__name__, "n_iter": model_n_iter(model)} for name, model in shadow_models.items()},
        },
        "thresholds": {"core": list(core_thresholds), "shadow": list(shadow_thresholds)},
        "best_candidate": best_candidate,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_order": ["source_ge2", "predicted_2bpp", "target_2bpp"],
        "candidates": search_payloads,
        "interpretation_notes": [
            "This is a lightweight convolution-feature probe, not a deep CNN.",
            "Compare against Stage28 ge2 learned CJK visual 0.9684 before deciding whether heavier convolution is worth it.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TargetConvExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(target_glyphs),
        cjk_glyph_count=len(cjk_records),
        best_candidate=best_candidate,
        best_cjk_visual_score=float(best_payload["groups"]["cjk"]["visual_score"]),
    )
