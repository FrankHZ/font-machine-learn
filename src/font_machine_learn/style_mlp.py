from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import image_to_mask, mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import SOURCE_METADATA, STYLE_MLP_CONTACT, STYLE_MLP_METADATA, STAGE15_STYLE_MLP, TARGET_METADATA
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import compare_visual


SOURCE_MODES = ("visible", "ge2", "eq3")


@dataclass(frozen=True)
class StyleMlpExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    train_glyph_count: int
    best_mode: str
    best_cjk_visual_score: float
    visible_cjk_visual_score: float
    ge2_cjk_visual_score: float
    eq3_cjk_visual_score: float


def mask_pixel_features(mask: list[list[bool]], x: int, y: int) -> list[float]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    values: list[float] = []
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            sx = x + dx
            sy = y + dy
            values.append(1.0 if 0 <= sx < width and 0 <= sy < height and mask[sy][sx] else 0.0)

    neighbor_count = 0
    diagonal_count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sx = x + dx
            sy = y + dy
            if 0 <= sx < width and 0 <= sy < height and mask[sy][sx]:
                neighbor_count += 1
                if dx != 0 and dy != 0:
                    diagonal_count += 1
    nx = x / max(1, width - 1)
    ny = y / max(1, height - 1)
    values.extend(
        (
            nx,
            ny,
            min(nx, 1.0 - nx),
            min(ny, 1.0 - ny),
            neighbor_count / 8.0,
            diagonal_count / 4.0,
        )
    )
    return values


def glyph_training_rows(
    source_mask: list[list[bool]],
    target_levels: list[list[int]],
) -> tuple[list[list[float]], list[int]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for y, row in enumerate(target_levels):
        for x, value in enumerate(row):
            rows.append(mask_pixel_features(source_mask, x, y))
            labels.append(value)
    return rows, labels


def levels_from_predictions(predictions: np.ndarray, width: int, height: int) -> list[list[int]]:
    values = [int(value) for value in predictions.tolist()]
    return [values[row * width : (row + 1) * width] for row in range(height)]


def mean_visual(records: list, attr: str) -> float:
    if not records:
        return 0.0
    return sum(float(getattr(record, attr)) for record in records) / len(records)


def target_source_mask(glyph: dict, mode: str) -> list[list[bool]]:
    target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
    return levels_to_threshold_mask(target_levels, mode)


def train_mode_model(
    glyphs: list[dict],
    *,
    mode: str,
    train_glyph_count: int,
    hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> tuple[MLPClassifier, float]:
    train_rows: list[list[float]] = []
    train_labels: list[int] = []
    for glyph in glyphs[:train_glyph_count]:
        target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        source_mask = levels_to_threshold_mask(target_levels, mode)
        rows, labels = glyph_training_rows(source_mask, target_levels)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model, float(accuracy_score(y_train, model.predict(x_train)))


def predict_levels(model: MLPClassifier, source_mask: list[list[bool]], width: int, height: int) -> list[list[int]]:
    features = np.asarray(
        [mask_pixel_features(source_mask, x, y) for y in range(height) for x in range(width)],
        dtype=np.float32,
    )
    return levels_from_predictions(model.predict(features), width, height)


def export_style_mlp(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE15_STYLE_MLP,
    metadata_json: Path = STYLE_MLP_METADATA,
    contact_sheet: Path = STYLE_MLP_CONTACT,
    max_train_glyphs: int = 768,
    hidden_units: int = 64,
    max_iter: int = 60,
    random_seed: int = 15,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
    external_sources: dict[str, Path] | None = None,
) -> StyleMlpExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    train_glyph_count = min(max_train_glyphs, len(glyphs))

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    mode_payloads: dict[str, dict] = {}
    mode_models: dict[str, MLPClassifier] = {}
    best_records: list[BaselineGlyphMetrics] = []
    for mode in SOURCE_MODES:
        mode_dir = out_dir / mode
        source_dir = mode_dir / "source_1bpp"
        pred_dir = mode_dir / "predicted_2bpp"
        source_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        model, train_accuracy = train_mode_model(
            glyphs,
            mode=mode,
            train_glyph_count=train_glyph_count,
            hidden_units=hidden_units,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        mode_models[mode] = model

        visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
        records: list[BaselineGlyphMetrics] = []
        glyph_payloads: list[dict] = []
        for glyph in glyphs:
            index = int(glyph["index"])
            chars = list(glyph["chars"])
            char_class = classify_glyph(chars)
            target_levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
            source_mask = levels_to_threshold_mask(target_levels, mode)
            source_png = source_dir / f"glyph_{index:04d}.png"
            mask_to_image(source_mask).save(source_png)
            predicted_levels = predict_levels(model, source_mask, cell_width, cell_height)
            predicted_png = pred_dir / f"glyph_{index:04d}.png"
            levels_to_image(predicted_levels).save(predicted_png)
            visual = compare_visual(predicted_levels, target_levels)
            for group in ("all", group_name(char_class)):
                visual_by_group[group].append(visual)
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
            glyph_payloads.append(
                {
                    **asdict(record),
                    "char_class": char_class,
                    "visual": visual.to_dict(),
                }
            )

        groups = {
            name: {
                "glyph_count": len(visual_by_group[name]),
                "visual_score": mean_visual(visual_by_group[name], "visual_score"),
                "ink_f1": mean_visual(visual_by_group[name], "ink_f1"),
                "shadow_f1": mean_visual(visual_by_group[name], "shadow_f1"),
                "foreground_iou": mean_visual(visual_by_group[name], "foreground_iou"),
                "pixel_accuracy": mean_visual(visual_by_group[name], "pixel_accuracy"),
                "mean_absolute_error": mean_visual(visual_by_group[name], "mean_absolute_error"),
            }
            for name in ("all", "cjk", "non_cjk")
        }
        mode_payloads[mode] = {
            "mode": mode,
            "train_pixel_accuracy": train_accuracy,
            "model": {
                "kind": "sklearn.neural_network.MLPClassifier",
                "features": "5x5 source mask patch + normalized coordinates + neighbor counts",
                "hidden_units": hidden_units,
                "max_iter": max_iter,
                "random_seed": random_seed,
                "n_iter": int(model.n_iter_),
            },
            "source_dir": str(source_dir),
            "predicted_dir": str(pred_dir),
            "groups": groups,
            "glyphs": glyph_payloads,
            "_records": records,
        }

    best_mode = max(SOURCE_MODES, key=lambda mode: float(mode_payloads[mode]["groups"]["cjk"]["visual_score"]))
    best_records = mode_payloads[best_mode].pop("_records")
    for mode in SOURCE_MODES:
        mode_payloads[mode].pop("_records", None)

    _make_contact_sheet(
        best_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    external_payload: dict[str, dict] = {}
    source_candidates = external_sources
    if source_candidates is None:
        source_candidates = {"wqy13": SOURCE_METADATA}
        size14 = Path("data/processed/glyphs/stage14_wqy_size14/source_metadata.json")
        if size14.exists():
            source_candidates["wqy14"] = size14
    best_model = mode_models[best_mode]
    for name, source_path in source_candidates.items():
        if not source_path.exists():
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_glyphs = list(source["glyphs"])
        pred_dir = out_dir / f"external_{name}_{best_mode}" / "predicted_2bpp"
        pred_dir.mkdir(parents=True, exist_ok=True)
        external_contact = out_dir / f"external_{name}_{best_mode}" / "contact.png"
        visual_by_group = {"all": [], "cjk": [], "non_cjk": []}
        records: list[BaselineGlyphMetrics] = []
        for glyph in source_glyphs:
            index = int(glyph["index"])
            chars = list(glyph["chars"])
            char_class = classify_glyph(chars)
            source_mask = image_to_mask(glyph["source_png"])
            target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
            predicted_levels = predict_levels(best_model, source_mask, cell_width, cell_height)
            predicted_png = pred_dir / f"glyph_{index:04d}.png"
            levels_to_image(predicted_levels).save(predicted_png)
            visual = compare_visual(predicted_levels, target_levels)
            for group in ("all", group_name(char_class)):
                visual_by_group[group].append(visual)
            records.append(
                BaselineGlyphMetrics(
                    index=index,
                    codes=list(glyph["codes"]),
                    chars=chars,
                    source_png=str(glyph["source_png"]),
                    predicted_png=str(predicted_png),
                    target_png=str(glyph["target_png"]),
                    pixel_accuracy=visual.pixel_accuracy,
                    mean_absolute_error=visual.mean_absolute_error,
                    foreground_iou=visual.foreground_iou,
                )
            )
        _make_contact_sheet(
            records,
            out_path=external_contact,
            scale=scale,
            columns=columns,
            cell_width=cell_width,
            cell_height=cell_height,
            pad=pad,
        )
        external_payload[name] = {
            "source_metadata": str(source_path),
            "model_mode": best_mode,
            "predicted_dir": str(pred_dir),
            "contact_sheet": str(external_contact),
            "groups": {
                group: {
                    "glyph_count": len(visual_by_group[group]),
                    "visual_score": mean_visual(visual_by_group[group], "visual_score"),
                    "ink_f1": mean_visual(visual_by_group[group], "ink_f1"),
                    "shadow_f1": mean_visual(visual_by_group[group], "shadow_f1"),
                    "foreground_iou": mean_visual(visual_by_group[group], "foreground_iou"),
                    "pixel_accuracy": mean_visual(visual_by_group[group], "pixel_accuracy"),
                    "mean_absolute_error": mean_visual(visual_by_group[group], "mean_absolute_error"),
                }
                for group in ("all", "cjk", "non_cjk")
            },
        }

    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(glyphs),
        "train_glyph_count": train_glyph_count,
        "task": "target-derived 1bpp mask -> NFTR 2bpp style levels",
        "source_modes": {
            "visible": "target level > 0",
            "ge2": "target level >= 2",
            "eq3": "target level == 3",
        },
        "best_mode": best_mode,
        "modes": mode_payloads,
        "contact_sheet": str(contact_sheet),
        "contact_sheet_mode": best_mode,
        "external_sources": external_payload,
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return StyleMlpExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        glyph_count=len(glyphs),
        train_glyph_count=train_glyph_count,
        best_mode=best_mode,
        best_cjk_visual_score=float(mode_payloads[best_mode]["groups"]["cjk"]["visual_score"]),
        visible_cjk_visual_score=float(mode_payloads["visible"]["groups"]["cjk"]["visual_score"]),
        ge2_cjk_visual_score=float(mode_payloads["ge2"]["groups"]["cjk"]["visual_score"]),
        eq3_cjk_visual_score=float(mode_payloads["eq3"]["groups"]["cjk"]["visual_score"]),
    )
