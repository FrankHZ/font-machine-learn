from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.boundary_rules import BoundaryRule, boundary_levels
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.nftr import checkerboard, export_target_dataset, palette_rgba
from font_machine_learn.paths import (
    PATCH_READINESS_CONTACT,
    PATCH_READINESS_METADATA,
    PATCH_READINESS_PATCHES,
    STAGE15_STYLE_MLP,
    TARGET_METADATA,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import compare_visual


LevelGrid = list[list[int]]
Mask = list[list[bool]]


@dataclass(frozen=True)
class PatchReadinessExport:
    target_metadata: str
    mlp_pred_dir: str
    metadata_json: str
    patches_jsonl: str
    contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    patch_record_count: int
    rule_wrong_mlp_right: int
    mlp_wrong_rule_right: int
    both_wrong: int


def _read_levels(path: Path) -> LevelGrid:
    return image_to_target_levels(Image.open(path).convert("RGBA"))


def _load_mlp_levels(index: int, fallback: LevelGrid, mlp_pred_dir: Path) -> tuple[LevelGrid, bool]:
    path = mlp_pred_dir / f"glyph_{index:04d}.png"
    if path.exists():
        return _read_levels(path), True
    return [row[:] for row in fallback], False


def _mask_patch_bits(mask: Mask, x: int, y: int, radius: int) -> str:
    height = len(mask)
    width = len(mask[0]) if height else 0
    bits: list[str] = []
    for ty in range(y - radius, y + radius + 1):
        for tx in range(x - radius, x + radius + 1):
            bits.append("1" if 0 <= ty < height and 0 <= tx < width and mask[ty][tx] else "0")
    return "".join(bits)


def _levels_patch(levels: LevelGrid, x: int, y: int, radius: int) -> LevelGrid:
    height = len(levels)
    width = len(levels[0]) if height else 0
    patch: LevelGrid = []
    for ty in range(y - radius, y + radius + 1):
        row: list[int] = []
        for tx in range(x - radius, x + radius + 1):
            row.append(levels[ty][tx] if 0 <= ty < height and 0 <= tx < width else 0)
        patch.append(row)
    return patch


def _mask_patch_to_levels(mask: Mask, x: int, y: int, radius: int) -> LevelGrid:
    height = len(mask)
    width = len(mask[0]) if height else 0
    patch: LevelGrid = []
    for ty in range(y - radius, y + radius + 1):
        row: list[int] = []
        for tx in range(x - radius, x + radius + 1):
            row.append(3 if 0 <= ty < height and 0 <= tx < width and mask[ty][tx] else 0)
        patch.append(row)
    return patch


def _category(target: int, mlp: int, rule: int) -> str:
    mlp_right = mlp == target
    rule_right = rule == target
    if mlp_right and rule_right:
        return "both_right"
    if mlp_right:
        return "rule_wrong_mlp_right"
    if rule_right:
        return "mlp_wrong_rule_right"
    return "both_wrong"


def _empty_counts() -> dict[str, int]:
    return {
        "both_right": 0,
        "rule_wrong_mlp_right": 0,
        "mlp_wrong_rule_right": 0,
        "both_wrong": 0,
    }


def _confusion() -> dict[str, dict[str, int]]:
    return {str(actual): {str(predicted): 0 for predicted in range(4)} for actual in range(4)}


def _bump_confusion(matrix: dict[str, dict[str, int]], actual: int, predicted: int) -> None:
    matrix[str(actual)][str(predicted)] += 1


def _summarize_confusion(matrix: dict[str, dict[str, int]]) -> dict:
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[str(level)][str(level)] for level in range(4))
    return {
        "total": total,
        "accuracy": 0.0 if total == 0 else correct / total,
        "matrix": matrix,
    }


def _mark_center(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    cx = image.width // 2
    cy = image.height // 2
    draw.rectangle((cx - 1, cy - 1, cx + 1, cy + 1), outline=(220, 34, 34, 255))


def _make_patch_contact(
    examples: list[dict],
    *,
    out_path: Path,
    radius: int,
    scale: int,
    columns: int,
    pad: int,
) -> None:
    patch_size = radius * 2 + 1
    tile = patch_size * scale
    group_h = tile * 4 + pad * 3
    rows = (len(examples) + columns - 1) // columns
    sheet = checkerboard((columns * tile + (columns + 1) * pad, rows * group_h + (rows + 1) * pad), max(2, scale))
    for position, example in enumerate(examples):
        x0 = pad + (position % columns) * (tile + pad)
        y0 = pad + (position // columns) * (group_h + pad)
        patches = (
            example["source_patch_levels"],
            example["target_patch_levels"],
            example["mlp_patch_levels"],
            example["rule_patch_levels"],
        )
        for row_index, patch in enumerate(patches):
            image = levels_to_image(patch).resize((tile, tile), Image.Resampling.NEAREST)
            _mark_center(image)
            sheet.alpha_composite(image, (x0, y0 + row_index * (tile + pad)))
    sheet.save(out_path)


def export_patch_readiness(
    target_metadata: Path = TARGET_METADATA,
    *,
    mlp_pred_dir: Path = STAGE15_STYLE_MLP / "ge2" / "predicted_2bpp",
    metadata_json: Path = PATCH_READINESS_METADATA,
    patches_jsonl: Path = PATCH_READINESS_PATCHES,
    contact_sheet: Path = PATCH_READINESS_CONTACT,
    rule: BoundaryRule = BoundaryRule("n2_band0_plain", 2, 0, False),
    patch_radius: int = 4,
    examples_per_category: int = 48,
    contact_scale: int = 6,
    contact_columns: int = 16,
    pad: int = 1,
) -> PatchReadinessExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    glyphs = list(target["glyphs"])
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    patches_jsonl.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    groups = {
        "all": {"mlp": [], "rule": []},
        "cjk": {"mlp": [], "rule": []},
        "non_cjk": {"mlp": [], "rule": []},
    }
    ge2_confusion = {"mlp": _confusion(), "rule": _confusion()}
    class_counts = {
        "ge2_cjk": {"2": 0, "3": 0},
        "ge2_cjk_categories": {str(level): _empty_counts() for level in (2, 3)},
    }
    examples_by_category = {name: [] for name in _empty_counts()}
    cjk_glyph_count = 0
    patch_record_count = 0
    mlp_available_count = 0

    with patches_jsonl.open("w", encoding="utf-8") as writer:
        for glyph in glyphs:
            index = int(glyph["index"])
            chars = list(glyph["chars"])
            char_class = classify_glyph(chars)
            group = "cjk" if char_class == "cjk" else "non_cjk"
            if group == "cjk":
                cjk_glyph_count += 1

            target_levels = _read_levels(Path(glyph["png"]))
            source_mask = levels_to_threshold_mask(target_levels, "ge2")
            rule_levels = boundary_levels(target_levels, rule)
            mlp_levels, mlp_available = _load_mlp_levels(index, rule_levels, mlp_pred_dir)
            if mlp_available:
                mlp_available_count += 1

            mlp_metrics = compare_visual(mlp_levels, target_levels)
            rule_metrics = compare_visual(rule_levels, target_levels)
            for name in ("all", group):
                groups[name]["mlp"].append(mlp_metrics)
                groups[name]["rule"].append(rule_metrics)

            if group != "cjk":
                continue

            height = len(target_levels)
            width = len(target_levels[0]) if height else 0
            for y in range(height):
                for x in range(width):
                    target_value = target_levels[y][x]
                    if target_value not in (2, 3):
                        continue
                    class_counts["ge2_cjk"][str(target_value)] += 1
                    mlp_value = mlp_levels[y][x]
                    rule_value = rule_levels[y][x]
                    category = _category(target_value, mlp_value, rule_value)
                    class_counts["ge2_cjk_categories"][str(target_value)][category] += 1
                    _bump_confusion(ge2_confusion["mlp"], target_value, mlp_value)
                    _bump_confusion(ge2_confusion["rule"], target_value, rule_value)

                    record = {
                        "glyph_index": index,
                        "codes": list(glyph["codes"]),
                        "chars": chars,
                        "x": x,
                        "y": y,
                        "target": target_value,
                        "mlp": mlp_value,
                        "rule": rule_value,
                        "category": category,
                        "source_patch_bits": _mask_patch_bits(source_mask, x, y, patch_radius),
                    }
                    writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    patch_record_count += 1

                    bucket = examples_by_category[category]
                    if len(bucket) < examples_per_category:
                        bucket.append(
                            {
                                **record,
                                "source_patch_levels": _mask_patch_to_levels(source_mask, x, y, patch_radius),
                                "target_patch_levels": _levels_patch(target_levels, x, y, patch_radius),
                                "mlp_patch_levels": _levels_patch(mlp_levels, x, y, patch_radius),
                                "rule_patch_levels": _levels_patch(rule_levels, x, y, patch_radius),
                            }
                        )

    ordered_examples: list[dict] = []
    for name in ("rule_wrong_mlp_right", "mlp_wrong_rule_right", "both_wrong", "both_right"):
        ordered_examples.extend(examples_by_category[name])
    _make_patch_contact(
        ordered_examples,
        out_path=contact_sheet,
        radius=patch_radius,
        scale=contact_scale,
        columns=contact_columns,
        pad=pad,
    )

    def summarize_metric(records: list, attr: str) -> float:
        return 0.0 if not records else sum(float(getattr(record, attr)) for record in records) / len(records)

    metric_summary = {
        name: {
            model: {
                "glyph_count": len(records),
                "visual_score": summarize_metric(records, "visual_score"),
                "ink_f1": summarize_metric(records, "ink_f1"),
                "shadow_f1": summarize_metric(records, "shadow_f1"),
                "foreground_iou": summarize_metric(records, "foreground_iou"),
                "pixel_accuracy": summarize_metric(records, "pixel_accuracy"),
            }
            for model, records in by_model.items()
        }
        for name, by_model in groups.items()
    }
    metadata = {
        "target_metadata": str(target_metadata),
        "mlp_pred_dir": str(mlp_pred_dir),
        "mlp_prediction_available_glyphs": mlp_available_count,
        "glyph_count": len(glyphs),
        "cjk_glyph_count": cjk_glyph_count,
        "task": "patch-readiness analysis for ge2 1bpp source -> NFTR 2bpp level assignment",
        "source_mask": "target-derived ge2 mask, value >= 2",
        "rule": asdict(rule),
        "patch_radius": patch_radius,
        "patch_size": patch_radius * 2 + 1,
        "patches_jsonl": str(patches_jsonl),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_order": ["source_ge2_patch", "target_patch", "mlp_patch", "rule_patch"],
        "contact_sheet_category_order": ["rule_wrong_mlp_right", "mlp_wrong_rule_right", "both_wrong", "both_right"],
        "patch_record_count": patch_record_count,
        "groups": metric_summary,
        "ge2_cjk_confusion": {
            "mlp": _summarize_confusion(ge2_confusion["mlp"]),
            "rule": _summarize_confusion(ge2_confusion["rule"]),
        },
        "class_counts": class_counts,
        "example_counts": {name: len(rows) for name, rows in examples_by_category.items()},
        "interpretation_notes": [
            "The JSONL contains only CJK target level 2/3 pixels because Stage 17 showed the remaining gap is mostly the edge/core split.",
            "rule_wrong_mlp_right is the key evidence bucket for whether a learned patch model is capturing useful context beyond a simple boundary rule.",
            "If mlp_wrong_rule_right or both_wrong dominates, the next stage should inspect labels/source masks before increasing model capacity.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    category_totals = _empty_counts()
    for value_counts in class_counts["ge2_cjk_categories"].values():
        for name, count in value_counts.items():
            category_totals[name] += count

    return PatchReadinessExport(
        target_metadata=str(target_metadata),
        mlp_pred_dir=str(mlp_pred_dir),
        metadata_json=str(metadata_json),
        patches_jsonl=str(patches_jsonl),
        contact_sheet=str(contact_sheet),
        glyph_count=len(glyphs),
        cjk_glyph_count=cjk_glyph_count,
        patch_record_count=patch_record_count,
        rule_wrong_mlp_right=category_totals["rule_wrong_mlp_right"],
        mlp_wrong_rule_right=category_totals["mlp_wrong_rule_right"],
        both_wrong=category_totals["both_wrong"],
    )
