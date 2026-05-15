from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.char_class import classify_glyph
from font_machine_learn.external_eval import summarize_visual
from font_machine_learn.nftr import checkerboard, export_target_dataset
from font_machine_learn.paths import (
    STAGE30_TARGET_TORCH,
    TARGET_METADATA,
    TARGET_TORCH_CONTACT,
    TARGET_TORCH_ERROR_CONTACT,
    TARGET_TORCH_METADATA,
)
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.visual_metrics import VisualMetrics, compare_visual


@dataclass(frozen=True)
class TargetTorchExport:
    target_metadata: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    error_contact_sheet: str
    glyph_count: int
    cjk_glyph_count: int
    cjk_visual_score: float
    epochs: int
    final_loss: float


class TinyGlyphCNN(nn.Module):
    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_training_tensors(target_glyphs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for glyph in target_glyphs:
        levels = image_to_target_levels(Image.open(glyph["png"]).convert("RGBA"))
        mask = levels_to_threshold_mask(levels, "ge2")
        inputs.append(np.asarray(mask, dtype=np.float32)[None, :, :])
        labels.append(np.asarray(levels, dtype=np.int64))
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(labels))


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels.reshape(-1), minlength=4).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.mean()


def predict_levels(model: nn.Module, source: torch.Tensor) -> list[list[int]]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(source.unsqueeze(0).to(device))[0].cpu().numpy()
    source_mask = source[0].cpu().numpy() > 0.5
    height, width = source_mask.shape
    predicted = np.zeros((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            if source_mask[y, x]:
                predicted[y, x] = 3 if logits[3, y, x] >= logits[2, y, x] else 2
            else:
                predicted[y, x] = 1 if logits[1, y, x] >= logits[0, y, x] else 0
    return predicted.tolist()


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


def export_target_torch_cnn(
    target_metadata: Path = TARGET_METADATA,
    *,
    out_dir: Path = STAGE30_TARGET_TORCH,
    metadata_json: Path = TARGET_TORCH_METADATA,
    contact_sheet: Path = TARGET_TORCH_CONTACT,
    error_contact_sheet: Path = TARGET_TORCH_ERROR_CONTACT,
    channels: int = 48,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 0.003,
    random_seed: int = 30,
    contact_count: int = 160,
    worst_count: int = 160,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> TargetTorchExport:
    if not target_metadata.exists():
        export_target_dataset(Path("a.NFTR"))
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    target_glyphs = list(target["glyphs"])
    cell_width = int(target["cell_width"])
    cell_height = int(target["cell_height"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    error_contact_sheet.parent.mkdir(parents=True, exist_ok=True)

    inputs, labels = make_training_tensors(target_glyphs)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

    source_dir = out_dir / "source_ge2"
    pred_dir = out_dir / "predicted_2bpp"
    source_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    visual_by_group: dict[str, list[VisualMetrics]] = {"all": [], "cjk": [], "non_cjk": []}
    records: list[dict] = []
    for tensor_index, glyph in enumerate(target_glyphs):
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char_class = classify_glyph(chars)
        group = "cjk" if char_class == "cjk" else "non_cjk"
        target_levels = labels[tensor_index].numpy().astype(np.int64).tolist()
        source_levels = [[2 if value else 0 for value in row] for row in inputs[tensor_index, 0].numpy().astype(bool).tolist()]
        predicted_levels = predict_levels(model, inputs[tensor_index])
        visual = compare_visual(predicted_levels, target_levels)
        for key in ("all", group):
            visual_by_group[key].append(visual)

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
                "target_png": str(glyph["png"]),
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

    groups = {key: summarize_visual(visual_by_group[key]) for key in ("all", "cjk", "non_cjk")}
    payload = {
        "target_metadata": str(target_metadata),
        "glyph_count": len(target_glyphs),
        "cjk_glyph_count": len(cjk_records),
        "task": "tiny PyTorch CNN: target >=2 1bpp mask -> target 2bpp levels",
        "source_mask": "target level >= 2",
        "model": {
            "kind": "TinyGlyphCNN",
            "channels": channels,
            "layers": "3x Conv3x3 ReLU blocks + Conv1x1 to 4 classes",
            "inference_constraint": "source pixels are forced to levels 2/3; non-source pixels are forced to levels 0/1",
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "losses": losses,
            "final_loss": losses[-1] if losses else 0.0,
            "class_weights": class_weights(labels).tolist(),
        },
        "groups": groups,
        "contact_sheet": str(contact_sheet),
        "error_contact_sheet": str(error_contact_sheet),
        "contact_sheet_order": ["source_ge2", "predicted_2bpp", "target_2bpp"],
        "glyphs": records,
        "interpretation_notes": [
            "This trains and evaluates on the target-shaped source mask as a capacity probe.",
            "Compare against Stage28 ge2 learned CJK visual 0.9684 and Stage29 lightweight conv 0.9604.",
        ],
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return TargetTorchExport(
        target_metadata=str(target_metadata),
        out_dir=str(out_dir),
        metadata_json=str(metadata_json),
        contact_sheet=str(contact_sheet),
        error_contact_sheet=str(error_contact_sheet),
        glyph_count=len(target_glyphs),
        cjk_glyph_count=len(cjk_records),
        cjk_visual_score=float(groups["cjk"]["visual_score"]),
        epochs=epochs,
        final_loss=float(losses[-1] if losses else 0.0),
    )
