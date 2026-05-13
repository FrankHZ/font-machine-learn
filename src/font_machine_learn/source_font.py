from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from font_machine_learn.nftr import checkerboard, export_target_dataset


@dataclass(frozen=True)
class SourceGlyphRecord:
    index: int
    codes: list[int]
    chars: list[str]
    source_png: str
    target_png: str
    bbox: tuple[int, int, int, int] | None
    ink_bbox: tuple[int, int, int, int] | None
    ink_width: int
    ink_height: int
    target_width: dict[str, int]


@dataclass(frozen=True)
class SourceDatasetExport:
    font_path: str
    font_name: tuple[str, str]
    font_index: int
    font_size: int
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    rendered_count: int
    cell_width: int
    cell_height: int


def render_mask(
    font: ImageFont.FreeTypeFont,
    char: str,
    cell_width: int,
    cell_height: int,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    canvas = Image.new("L", (cell_width * 3, cell_height * 3), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), char, font=font)
    if bbox is None:
        return canvas.crop((0, 0, cell_width, cell_height)), None

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (cell_width - width) // 2 - bbox[0]
    y = (cell_height - height) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return canvas.crop((0, 0, cell_width, cell_height)), bbox


def quantize_mask_to_1bpp(mask: Image.Image, threshold: int = 96) -> Image.Image:
    out = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    pixels = []
    data = mask.get_flattened_data() if hasattr(mask, "get_flattened_data") else mask.getdata()
    for value in data:
        pixels.append((0, 0, 0, 255) if value >= threshold else (0, 0, 0, 0))
    out.putdata(pixels)
    return out


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = image.getchannel("A")
    return alpha.getbbox()


def first_char(chars: list[str]) -> str | None:
    for char in chars:
        if char:
            return char
    return None


def make_pair_contact_sheet(
    records: list[SourceGlyphRecord],
    *,
    source_dir: Path,
    scale: int,
    columns: int,
    cell_width: int,
    cell_height: int,
    pad: int,
) -> Image.Image:
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    pair_h = tile_h * 2 + pad
    rows = (len(records) + columns - 1) // columns
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, rows * pair_h + (rows + 1) * pad),
        max(2, scale * 2),
    )

    for record in records:
        x = pad + (record.index % columns) * (tile_w + pad)
        y = pad + (record.index // columns) * (pair_h + pad)
        source = Image.open(source_dir / Path(record.source_png).name).convert("RGBA")
        target = Image.open(record.target_png).convert("RGBA")
        sheet.alpha_composite(source.resize((tile_w, tile_h), Image.Resampling.NEAREST), (x, y))
        sheet.alpha_composite(target.resize((tile_w, tile_h), Image.Resampling.NEAREST), (x, y + tile_h + pad))
    return sheet


def export_source_dataset(
    nftr_source: Path = Path("a.NFTR"),
    font_path: Path = Path("wqy-zenhei.ttc"),
    out_dir: Path = Path("data/processed/glyphs/source"),
    *,
    target_metadata: Path | None = None,
    font_index: int = 2,
    font_size: int = 13,
    cell_width: int = 15,
    cell_height: int = 15,
    threshold: int = 96,
    metadata_json: Path | None = None,
    contact_sheet: Path | None = None,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> SourceDatasetExport:
    target_metadata_path = target_metadata or Path("data/processed/glyphs/target_metadata.json")
    if not target_metadata_path.exists():
        export_target_dataset(nftr_source)

    target = json.loads(target_metadata_path.read_text(encoding="utf-8"))
    font = ImageFont.truetype(str(font_path), size=font_size, index=font_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_json or out_dir.parent / "source_metadata.json"
    contact_path = contact_sheet or out_dir.parent / "source_target_contact.png"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[SourceGlyphRecord] = []
    for glyph in target["glyphs"]:
        index = int(glyph["index"])
        chars = list(glyph["chars"])
        char = first_char(chars)
        filename = f"glyph_{index:04d}.png"
        source_png = out_dir / filename

        bbox = None
        ink = None
        image = Image.new("RGBA", (cell_width, cell_height), (0, 0, 0, 0))
        if char is not None:
            mask, bbox = render_mask(font, char, cell_width, cell_height)
            image = quantize_mask_to_1bpp(mask, threshold)
            ink = alpha_bbox(image)
        image.save(source_png)

        if ink is None:
            ink_width = 0
            ink_height = 0
        else:
            ink_width = ink[2] - ink[0]
            ink_height = ink[3] - ink[1]

        records.append(
            SourceGlyphRecord(
                index=index,
                codes=list(glyph["codes"]),
                chars=chars,
                source_png=str(source_png),
                target_png=str(glyph["png"]),
                bbox=bbox,
                ink_bbox=ink,
                ink_width=ink_width,
                ink_height=ink_height,
                target_width=dict(glyph["width"]),
            )
        )

    contact = make_pair_contact_sheet(
        records,
        source_dir=out_dir,
        scale=scale,
        columns=columns,
        cell_width=cell_width,
        cell_height=cell_height,
        pad=pad,
    )
    contact.save(contact_path)

    payload = {
        "font_path": str(font_path),
        "font_name": font.getname(),
        "font_index": font_index,
        "font_size": font_size,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "threshold": threshold,
        "glyph_count": len(records),
        "rendered_count": sum(1 for record in records if record.chars),
        "glyph_dir": str(out_dir),
        "target_metadata": str(target_metadata_path),
        "contact_sheet": str(contact_path),
        "glyphs": [asdict(record) for record in records],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return SourceDatasetExport(
        font_path=str(font_path),
        font_name=font.getname(),
        font_index=font_index,
        font_size=font_size,
        out_dir=str(out_dir),
        metadata_json=str(metadata_path),
        contact_sheet=str(contact_path),
        glyph_count=len(records),
        rendered_count=payload["rendered_count"],
        cell_width=cell_width,
        cell_height=cell_height,
    )
