from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.binary_diagnostic import compare_masks, mask_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.external_eval import summarize_binary, summarize_visual
from font_machine_learn.nftr import checkerboard, export_target_dataset
from font_machine_learn.paths import STAGE28_TARGET_QUANTIZED, TARGET_METADATA, TARGET_QUANTIZED_METADATA
from font_machine_learn.song13_layer_mlp import (
    build_target_ge2_training_rows,
    predict_source_locked_levels,
    train_binary_models,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


Mask = list[list[bool]]
LevelGrid = list[list[int]]
MODES = ("visible", "ge2", "eq3")


@dataclass(frozen=True)
class TargetQuantizedCalibrationExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    glyph_count: int
    cjk_glyph_count: int
    best_mode_by_learned_cjk_visual: str
    best_learned_cjk_visual_score: float


def mask_to_flat_levels(mask: Mask, *, level: int = 3) -> LevelGrid:
    return [[level if value else 0 for value in row] for row in mask]


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _group(char_class: str) -> str:
    return "cjk" if char_class == "cjk" else "non_cjk"


def train_stage26_style_heads(
    target_glyphs: list[dict],
    *,
    patch_radius: int,
    max_train_glyphs: int | None,
    core_hidden_units: int,
    shadow_hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> tuple[object, object, dict]:
    x_core, y_core, x_shadow, y_shadow, train_cjk = build_target_ge2_training_rows(
        target_glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
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
    return (
        core_models["core_patch_mlp"],
        shadow_models["shadow_logistic_balanced"],
        {
            "train_cjk_glyph_count": train_cjk,
            "train_pixel_count": {"core_edge": int(len(y_core)), "shadow": int(len(y_shadow))},
            "label_counts": {
                "core_edge": {str(value): int(np.count_nonzero(y_core == value)) for value in (0, 1)},
                "shadow": {str(value): int(np.count_nonzero(y_shadow == value)) for value in (0, 1)},
            },
            "models": {
                "core_edge": sorted(core_models),
                "shadow": sorted(shadow_models),
                "selected": {
                    "core_edge": "core_patch_mlp",
                    "shadow": "shadow_logistic_balanced",
                },
            },
        },
    )


def make_mode_contact_sheet(
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
    group_h = tile_h * 4 + pad * 3
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, sheet_rows * group_h + (sheet_rows + 1) * pad),
        max(2, scale * 2),
    )
    keys = ("source_png", "flat3_png", "learned_png", "target_png")
    for position, row in enumerate(rows):
        x = pad + (position % columns) * (tile_w + pad)
        y = pad + (position // columns) * (group_h + pad)
        for image_index, key in enumerate(keys):
            image = Image.open(row[key]).convert("RGBA")
            scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
            sheet.alpha_composite(scaled, (x, y + image_index * (tile_h + pad)))
    sheet.save(out_path)


def evaluate_quantized_mode(
    mode: str,
    target_glyphs: list[dict],
    *,
    out_dir: Path,
    core_model: object,
    shadow_model: object,
    patch_radius: int,
    core_threshold: float,
    shadow_threshold: float,
    cell_width: int,
    cell_height: int,
    contact_count: int,
    scale: int,
    columns: int,
    pad: int,
) -> dict:
    mode_dir = out_dir / mode
    source_dir = mode_dir / "source_mask"
    flat3_dir = mode_dir / "flat3"
    learned_dir = mode_dir / "learned_2bpp"
    for path in (source_dir, flat3_dir, learned_dir):
        path.mkdir(parents=True, exist_ok=True)

    visual_by_model: dict[str, dict[str, list[VisualMetrics]]] = {
        "flat3": {"all": [], "cjk": [], "non_cjk": []},
        "learned": {"all": [], "cjk": [], "non_cjk": []},
    }
    mask_metrics = {"all": [], "cjk": [], "non_cjk": []}
    source_foreground_ratios = {"all": [], "cjk": [], "non_cjk": []}
    records: list[dict] = []

    for glyph in target_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = _group(char_class)
        target_png = str(glyph["png"])
        target_levels = image_to_target_levels(Image.open(target_png).convert("RGBA"))
        source_mask = levels_to_threshold_mask(target_levels, mode)
        target_visible = levels_to_threshold_mask(target_levels, "visible")
        flat3_levels = mask_to_flat_levels(source_mask, level=3)
        learned_levels = predict_source_locked_levels(
            core_model,
            shadow_model,
            source_mask,
            patch_radius=patch_radius,
            core_threshold=core_threshold,
            shadow_threshold=shadow_threshold,
        )

        source_png = source_dir / f"glyph_{index:04d}.png"
        flat3_png = flat3_dir / f"glyph_{index:04d}.png"
        learned_png = learned_dir / f"glyph_{index:04d}.png"
        mask_to_image(source_mask).save(source_png)
        levels_to_image(flat3_levels).save(flat3_png)
        levels_to_image(learned_levels).save(learned_png)

        source_metric = compare_masks(source_mask, target_visible)
        flat3_visual = compare_visual(flat3_levels, target_levels)
        learned_visual = compare_visual(learned_levels, target_levels)
        for key in ("all", group):
            mask_metrics[key].append(source_metric)
            source_foreground_ratios[key].append(
                sum(1 for row in source_mask for value in row if value) / (cell_width * cell_height)
            )
            visual_by_model["flat3"][key].append(flat3_visual)
            visual_by_model["learned"][key].append(learned_visual)

        records.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": chars,
                "char_class": char_class,
                "target_png": target_png,
                "source_png": str(source_png),
                "flat3_png": str(flat3_png),
                "learned_png": str(learned_png),
                "source_mask_vs_target_visible": asdict(source_metric),
                "flat3": flat3_visual.to_dict(),
                "learned": learned_visual.to_dict(),
            }
        )

    contact_records = [row for row in records if row["char_class"] == "cjk"][:contact_count]
    contact_sheet = mode_dir / f"target_quantized_{mode}_contact.png"
    make_mode_contact_sheet(
        contact_records,
        out_path=contact_sheet,
        cell_width=cell_width,
        cell_height=cell_height,
        scale=scale,
        columns=columns,
        pad=pad,
    )
    return {
        "mode": mode,
        "source_mask": {
            "definition": {
                "visible": "target level > 0",
                "ge2": "target level >= 2",
                "eq3": "target level == 3",
            }[mode],
            "dir": str(source_dir),
            "vs_target_visible": {key: summarize_binary(mask_metrics[key]) for key in ("all", "cjk", "non_cjk")},
            "foreground_ratio": {key: _mean(source_foreground_ratios[key]) for key in ("all", "cjk", "non_cjk")},
        },
        "outputs": {
            "flat3": {
                "dir": str(flat3_dir),
                "meaning": "paint every source pixel as level 3; no learned layer recovery",
                "groups": {key: summarize_visual(visual_by_model["flat3"][key]) for key in ("all", "cjk", "non_cjk")},
            },
            "learned": {
                "dir": str(learned_dir),
                "meaning": "Stage26 source-locked layer recovery from the quantized source mask",
                "groups": {key: summarize_visual(visual_by_model["learned"][key]) for key in ("all", "cjk", "non_cjk")},
            },
        },
        "contact_sheet": str(contact_sheet),
        "contact_sheet_order": ["source_mask", "flat3", "learned_2bpp", "target_2bpp"],
        "glyphs": records,
    }


def export_target_quantized_calibration(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE28_TARGET_QUANTIZED,
    metadata_json: Path = TARGET_QUANTIZED_METADATA,
    modes: tuple[str, ...] = MODES,
    patch_radius: int = 4,
    core_threshold: float = 0.55,
    shadow_threshold: float = 0.45,
    max_train_glyphs: int | None = None,
    core_hidden_units: int = 64,
    shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 28,
    contact_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> TargetQuantizedCalibrationExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    target_glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    core_model, shadow_model, train_payload = train_stage26_style_heads(
        target_glyphs,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
        core_hidden_units=core_hidden_units,
        shadow_hidden_units=shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )
    mode_payloads = {
        mode: evaluate_quantized_mode(
            mode,
            target_glyphs,
            out_dir=out_dir,
            core_model=core_model,
            shadow_model=shadow_model,
            patch_radius=patch_radius,
            core_threshold=core_threshold,
            shadow_threshold=shadow_threshold,
            cell_width=cell_width,
            cell_height=cell_height,
            contact_count=contact_count,
            scale=scale,
            columns=columns,
            pad=pad,
        )
        for mode in modes
    }
    best_mode = max(
        mode_payloads,
        key=lambda mode: float(mode_payloads[mode]["outputs"]["learned"]["groups"]["cjk"]["visual_score"]),
    )
    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(target_glyphs),
        "cjk_glyph_count": sum(1 for glyph in target_glyphs if classify_glyph(list(glyph["chars"])) == "cjk"),
        "task": "quantize target 2bpp glyphs to 1bpp source masks, then measure 2bpp recovery loss",
        "reason": "Calibrate the source-shape upper bound before trying convolutional models on Song13.",
        "patch_radius": patch_radius,
        "stage26_thresholds": {"core": core_threshold, "shadow": shadow_threshold},
        "train": train_payload,
        "best_mode_by_learned_cjk_visual": best_mode,
        "modes": mode_payloads,
        "interpretation_notes": [
            "`flat3` is the no-layer baseline from the quantized source mask.",
            "`learned` uses the same source-locked Stage26 style recovery as Song13, but with target-shaped source masks.",
            "`ge2` should usually be the useful source mask because level 2 carries edge information and level 1 is mostly shadow.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TargetQuantizedCalibrationExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        glyph_count=len(target_glyphs),
        cjk_glyph_count=int(payload["cjk_glyph_count"]),
        best_mode_by_learned_cjk_visual=best_mode,
        best_learned_cjk_visual_score=float(mode_payloads[best_mode]["outputs"]["learned"]["groups"]["cjk"]["visual_score"]),
    )

