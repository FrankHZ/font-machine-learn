from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.external_eval import summarize_visual
from font_machine_learn.nftr import export_target_dataset
from font_machine_learn.paths import (
    SONG13_TORCH_CONTACT,
    SONG13_TORCH_ERROR_CONTACT,
    SONG13_TORCH_METADATA,
    STAGE31_SONG13_TORCH,
    TARGET_METADATA,
)
from font_machine_learn.song13_adapter import DEFAULT_SONG13_SOURCE_METADATA
from font_machine_learn.song13_source_locked import source_deleted_ratio, source_level_ratios
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.target_torch_cnn import TinyGlyphCNN, class_weights, make_contact_sheet, make_training_tensors, predict_levels
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class Song13TorchExport:
    target_metadata: str
    source_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    cjk_visual_score: float
    cjk_source_deleted_ratio: float
    epochs: int
    final_loss: float


def source_png_to_tensor(path: Path) -> torch.Tensor:
    alpha = Image.open(path).convert("RGBA").getchannel("A")
    width, height = alpha.size
    values = np.zeros((1, height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if alpha.getpixel((x, y)):
                values[0, y, x] = 1.0
    return torch.from_numpy(values)


def source_tensor_to_levels(source: torch.Tensor) -> list[list[int]]:
    mask = source[0].numpy() > 0.5
    return [[2 if bool(value) else 0 for value in row] for row in mask.tolist()]


def train_target_ge2_cnn(
    target_glyphs: list[dict],
    *,
    channels: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
) -> tuple[nn.Module, list[float], torch.device, str]:
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = make_training_tensors(target_glyphs)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=True)
    model = TinyGlyphCNN(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
    losses: list[float] = []
    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * int(batch_x.shape[0])
            total_count += int(batch_x.shape[0])
        losses.append(total_loss / max(1, total_count))
    cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return model, losses, device, cuda_device


def export_song13_torch_cnn(
    target_metadata: Path = TARGET_METADATA,
    source_metadata: Path = DEFAULT_SONG13_SOURCE_METADATA,
    *,
    out_dir: Path = STAGE31_SONG13_TORCH,
    metadata_json: Path = SONG13_TORCH_METADATA,
    contact_sheet: Path = SONG13_TORCH_CONTACT,
    error_contact_sheet: Path = SONG13_TORCH_ERROR_CONTACT,
    channels: int = 48,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 0.003,
    random_seed: int = 31,
    eval_limit: int | None = None,
    contact_count: int = 160,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> Song13TorchExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    if not source_metadata.exists():
        raise FileNotFoundError(
            f"Source metadata not found: {source_metadata}. "
            "Render Song13 first with scripts/render_source_glyphs.py."
        )

    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    target_glyphs = list(target["glyphs"])
    source_glyphs = list(source["glyphs"])
    if eval_limit is not None:
        source_glyphs = source_glyphs[:eval_limit]
    cell_width = int(source["cell_width"])
    cell_height = int(source["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    model, losses, device, cuda_device = train_target_ge2_cnn(
        target_glyphs,
        channels=channels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    source_dir = out_dir / "source_ge2"
    pred_dir = out_dir / "predicted_2bpp"
    source_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    source_deleted_by_group: dict[str, list[float]] = {"all": [], "cjk": [], "non_cjk": []}
    source_level2_by_group: dict[str, list[float]] = {"all": [], "cjk": [], "non_cjk": []}
    source_level3_by_group: dict[str, list[float]] = {"all": [], "cjk": [], "non_cjk": []}
    records: list[dict] = []
    for glyph in source_glyphs:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = "cjk" if char_class == "cjk" else "non_cjk"
        source_tensor = source_png_to_tensor(Path(glyph["source_png"]))
        source_levels = source_tensor_to_levels(source_tensor)
        predicted_levels = predict_levels(model, source_tensor)
        target_levels = image_to_target_levels(Image.open(glyph["target_png"]).convert("RGBA"))
        source_mask = levels_to_threshold_mask(source_levels, "ge2")
        visual = compare_visual(predicted_levels, target_levels)
        deleted = source_deleted_ratio(source_mask, predicted_levels)
        level_ratios = source_level_ratios(source_mask, predicted_levels)
        for key in ("all", group):
            visual_by_group[key].append(visual)
            source_deleted_by_group[key].append(deleted)
            source_level2_by_group[key].append(float(level_ratios["level2"]))
            source_level3_by_group[key].append(float(level_ratios["level3"]))

        source_png = source_dir / f"glyph_{index:04d}.png"
        predicted_png = pred_dir / f"glyph_{index:04d}.png"
        levels_to_image(source_levels).save(source_png)
        levels_to_image(predicted_levels).save(predicted_png)
        records.append(
            {
                "index": index,
                "codes": list(glyph["codes"]),
                "chars": chars,
                "char_class": char_class,
                "source_png": str(source_png),
                "predicted_png": str(predicted_png),
                "target_png": str(glyph["target_png"]),
                "source_deleted_ratio": deleted,
                "source_level_ratios": level_ratios,
                "visual": visual.to_dict(),
            }
        )

    cjk_records = [record for record in records if record["char_class"] == "cjk"]
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

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    groups = {}
    for key in ("all", "cjk", "non_cjk"):
        groups[key] = summarize_visual(visual_by_group[key]) | {
            "source_deleted_ratio": mean(source_deleted_by_group[key]),
            "source_level2_ratio": mean(source_level2_by_group[key]),
            "source_level3_ratio": mean(source_level3_by_group[key]),
        }
    payload = {
        "target_metadata": str(target_metadata),
        "source_metadata": str(source_metadata),
        "glyph_count": len(records),
        "cjk_glyph_count": len(cjk_records),
        "eval_limit": eval_limit,
        "task": "transfer target-trained source-locked tiny PyTorch CNN to Song13 source masks",
        "source_mask": "Song13 1bpp rendered source",
        "model": {
            "kind": "TinyGlyphCNN",
            "channels": channels,
            "layers": "3x Conv3x3 ReLU blocks + Conv1x1 to 4 classes",
            "training_source": "target level >= 2 mask",
            "inference_constraint": "source pixels are forced to levels 2/3; non-source pixels are forced to levels 0/1",
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "device": str(device),
            "cuda_device": cuda_device,
            "losses": losses,
            "final_loss": losses[-1] if losses else 0.0,
        },
        "groups": groups,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_order": ["source_ge2", "predicted_2bpp", "target_2bpp"],
        "glyphs": records,
        "interpretation_notes": [
            "Pixel overlap remains a weak proxy for Song13 because its shape differs from the target NFTR.",
            "This stage is mainly a visual transfer check before building a full-map NFTR.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Song13TorchExport(
        target_metadata=str(target_metadata),
        source_metadata=str(source_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(records),
        cjk_glyph_count=len(cjk_records),
        cjk_visual_score=float(groups["cjk"]["visual_score"]),
        cjk_source_deleted_ratio=float(groups["cjk"]["source_deleted_ratio"]),
        epochs=epochs,
        final_loss=float(losses[-1] if losses else 0.0),
    )
