from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from font_machine_learn.baseline import BaselineGlyphMetrics, image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, image_to_mask
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.cjk_style import _make_contact_sheet, group_name
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.patch_classifier import build_training_rows, train_models
from font_machine_learn.paths import EXTERNAL_EVAL_METADATA, SOURCE_METADATA, STAGE21_EXTERNAL_EVAL, TARGET_METADATA
from font_machine_learn.shadow_classifier import (
    binary_accuracy,
    binary_f1_from_confusion,
    merge_binary_confusion,
    predict_combined_levels,
    shadow_confusion,
    shadow_training_rows,
    train_shadow_models,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class ExternalEvalExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    glyph_count: int
    train_core_edge_pixel_count: int
    train_shadow_pixel_count: int
    evaluated_sources: list[str]
    best_cjk_visual_score: float


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


def summarize_binary(records: list) -> dict[str, float | int]:
    if not records:
        return {
            "glyph_count": 0,
            "foreground_f1": 0.0,
            "foreground_iou": 0.0,
            "foreground_precision": 0.0,
            "foreground_recall": 0.0,
        }
    return {
        "glyph_count": len(records),
        "foreground_f1": sum(record.foreground_f1 for record in records) / len(records),
        "foreground_iou": sum(record.foreground_iou for record in records) / len(records),
        "foreground_precision": sum(record.foreground_precision for record in records) / len(records),
        "foreground_recall": sum(record.foreground_recall for record in records) / len(records),
    }


def default_external_sources() -> dict[str, Path]:
    sources = {"wqy13": SOURCE_METADATA}
    wqy14 = Path("data/processed/glyphs/stage14_wqy_size14/source_metadata.json")
    if wqy14.exists():
        sources["wqy14"] = wqy14
    song12 = Path("data/processed/glyphs/stage21_external_eval_sources/song12/source_metadata.json")
    if song12.exists():
        sources["song12"] = song12
    song13 = Path("data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json")
    if song13.exists():
        sources["song13"] = song13
    song14 = Path("data/processed/glyphs/stage21_external_eval_sources/song14/source_metadata.json")
    if song14.exists():
        sources["song14"] = song14
    return sources


def train_two_head_models(
    glyphs: list[dict],
    *,
    patch_radius: int,
    max_train_glyphs: int | None,
    core_hidden_units: int,
    shadow_hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> tuple[object, object, dict]:
    x_core, y_core, core_train_glyph_count = build_training_rows(
        glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    core_model = train_models(
        x_core,
        y_core,
        hidden_units=core_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )["patch_mlp"]
    x_shadow, y_shadow, shadow_train_glyph_count = shadow_training_rows(
        glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    shadow_model = train_shadow_models(
        x_shadow,
        y_shadow,
        hidden_units=shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )["shadow_patch_mlp"]
    return core_model, shadow_model, {
        "train_cjk_glyph_count": {
            "core_edge": core_train_glyph_count,
            "shadow": shadow_train_glyph_count,
        },
        "train_pixel_count": {
            "core_edge": int(len(y_core)),
            "shadow": int(len(y_shadow)),
        },
        "label_counts": {
            "core_edge": {str(value): int(np.count_nonzero(y_core == value)) for value in (2, 3)},
            "shadow": {str(value): int(np.count_nonzero(y_shadow == value)) for value in (0, 1)},
        },
    }


def evaluate_source(
    name: str,
    source_metadata: Path,
    *,
    out_dir: Path,
    core_model: object,
    shadow_model: object,
    patch_radius: int,
    scale: int,
    columns: int,
    pad: int,
    worst_count: int,
) -> dict:
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    glyphs = list(source["glyphs"])
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    pred_dir = out_dir / name / "predicted_2bpp"
    pred_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = out_dir / name / "external_contact.png"
    error_contact_sheet = out_dir / name / "external_error_contact.png"

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    source_mask_by_group = {"all": [], "cjk": [], "non_cjk": []}
    shadow_matrix = {str(actual): {str(value): 0 for value in (0, 1)} for actual in (0, 1)}
    records: list[BaselineGlyphMetrics] = []
    cjk_records: list[BaselineGlyphMetrics] = []
    visual_by_index: dict[int, float] = {}
    glyph_payloads: list[dict] = []
    for glyph in glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        source_mask = image_to_mask(glyph["source_png"])
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        target_ge2 = levels_to_threshold_mask(target_levels, "ge2")
        predicted_levels = predict_combined_levels(
            core_model,
            shadow_model,
            source_mask,
            patch_radius=patch_radius,
        )
        predicted_png = pred_dir / f"glyph_{index:04d}.png"
        levels_to_image(predicted_levels).save(predicted_png)
        visual = compare_visual(predicted_levels, target_levels)
        visual_by_index[index] = visual.visual_score
        mask_metrics = compare_masks(source_mask, target_ge2)
        for group in ("all", group_name(char_class)):
            visual_by_group[group].append(visual)
            source_mask_by_group[group].append(mask_metrics)
        if char_class == "cjk":
            merge_binary_confusion(shadow_matrix, shadow_confusion(predicted_levels, target_levels))
        record = BaselineGlyphMetrics(
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
        records.append(record)
        if char_class == "cjk":
            cjk_records.append(record)
        glyph_payloads.append({**asdict(record), "char_class": char_class, "visual": visual.to_dict()})

    _make_contact_sheet(
        cjk_records,
        out_path=contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    worst = sorted(cjk_records, key=lambda record: visual_by_index[record.index])[:worst_count]
    _make_contact_sheet(
        worst,
        out_path=error_contact_sheet,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )

    return {
        "source_metadata": str(source_metadata),
        "glyph_count": len(records),
        "cjk_glyph_count": len(cjk_records),
        "predicted_dir": str(pred_dir),
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "source_mask_vs_target_ge2": {
            group: summarize_binary(source_mask_by_group[group])
            for group in ("all", "cjk", "non_cjk")
        },
        "groups": {
            group: summarize_visual(visual_by_group[group])
            for group in ("all", "cjk", "non_cjk")
        },
        "shadow_cjk_confusion": {
            "accuracy": binary_accuracy(shadow_matrix),
            "foreground_f1": binary_f1_from_confusion(shadow_matrix),
            "matrix": shadow_matrix,
        },
        "glyphs": glyph_payloads,
    }


def export_external_eval(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE21_EXTERNAL_EVAL,
    metadata_json: Path = EXTERNAL_EVAL_METADATA,
    external_sources: dict[str, Path] | None = None,
    patch_radius: int = 4,
    max_train_glyphs: int | None = None,
    core_hidden_units: int = 64,
    shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 21,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> ExternalEvalExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    core_model, shadow_model, train_payload = train_two_head_models(
        glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
        core_hidden_units=core_hidden_units,
        shadow_hidden_units=shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )
    sources = external_sources if external_sources is not None else default_external_sources()
    source_payloads = {
        name: evaluate_source(
            name,
            source_path,
            out_dir=out_dir,
            core_model=core_model,
            shadow_model=shadow_model,
            patch_radius=patch_radius,
            scale=scale,
            columns=columns,
            pad=pad,
            worst_count=worst_count,
        )
        for name, source_path in sources.items()
        if source_path.exists()
    }
    best_cjk = max(
        (float(payload["groups"]["cjk"]["visual_score"]) for payload in source_payloads.values()),
        default=0.0,
    )
    metadata = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(glyphs),
        "task": "evaluate Stage20 two-head patch style model on external source masks",
        "source_mask_contract": "external source PNG alpha mask, usually WQY rendered 1bpp",
        "patch_radius": patch_radius,
        "patch_size": patch_radius * 2 + 1,
        "model": {
            "core_edge": "patch_mlp trained on target-derived ge2 level 2/3 pixels",
            "shadow": "shadow_patch_mlp trained on target-derived ge2 outside pixels",
        },
        **train_payload,
        "sources": source_payloads,
        "interpretation_notes": [
            "Controlled Stage20 score measures style learning with target-derived ge2 masks.",
            "This stage measures transfer when the input mask comes from rendered WQY source glyphs.",
            "Low WQY scores should be read primarily as source adaptation/alignment issues unless source_mask_vs_target_ge2 is already strong.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ExternalEvalExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        glyph_count=len(glyphs),
        train_core_edge_pixel_count=int(train_payload["train_pixel_count"]["core_edge"]),
        train_shadow_pixel_count=int(train_payload["train_pixel_count"]["shadow"]),
        evaluated_sources=list(source_payloads.keys()),
        best_cjk_visual_score=best_cjk,
    )
