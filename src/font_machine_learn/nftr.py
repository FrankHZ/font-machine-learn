from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

Layout = Literal["tiled8", "linear"]
NibbleOrder = Literal["high", "low"]


NITRO_MARKERS = (
    b"RTFN",
    b"FNIF",
    b"PLGC",
    b"HDWC",
    b"PAMC",
    b"NFTR",
    b"FINF",
    b"CGLP",
    b"CWDH",
    b"CMAP",
)

SECTION_FINF = b"FNIF"
SECTION_TGLP = b"PLGC"
SECTION_CWDH = b"HDWC"
SECTION_CMAP = b"PAMC"


@dataclass(frozen=True)
class ExportConfig:
    source: str
    out_png: str
    mode: str
    cell_width: int
    cell_height: int
    bpp: int
    layout: Layout
    nibble_order: NibbleOrder
    offset: int
    glyph_count: int
    columns: int
    scale: int
    pad: int
    bytes_per_glyph: int
    file_size: int
    markers_found: list[str]


def find_nitro_markers(data: bytes) -> list[str]:
    return [marker.decode("ascii") for marker in NITRO_MARKERS if marker in data]


def bytes_per_glyph(cell_width: int, cell_height: int, bpp: int) -> int:
    bits = cell_width * cell_height * bpp
    return math.ceil(bits / 8)


def parse_sections(data: bytes) -> list[tuple[int, bytes, int]]:
    if not data.startswith(b"RTFN"):
        return []

    sections = []
    off = 0x10
    while off + 8 <= len(data):
        magic = data[off : off + 4]
        size = struct.unpack_from("<I", data, off + 4)[0]
        if size < 8 or off + size > len(data):
            raise ValueError(f"invalid section at 0x{off:X}: {magic!r}, size={size}")
        sections.append((off, magic, size))
        off += size
    return sections


def choose_auto_offset(file_size: int, glyph_size: int) -> int:
    remainder = file_size % glyph_size
    if remainder == 0:
        return 0
    return remainder


def unpack_4bpp_byte(byte: int, nibble_order: NibbleOrder) -> tuple[int, int]:
    high = byte >> 4
    low = byte & 0x0F
    if nibble_order == "high":
        return high, low
    return low, high


def decode_linear_4bpp(
    payload: bytes,
    width: int,
    height: int,
    nibble_order: NibbleOrder,
) -> list[int]:
    pixels: list[int] = []
    for byte in payload:
        pixels.extend(unpack_4bpp_byte(byte, nibble_order))
    return pixels[: width * height]


def decode_linear_2bpp(payload: bytes, width: int, height: int) -> list[int]:
    pixels: list[int] = []
    for byte in payload:
        pixels.extend(((byte >> 6) & 0x03, (byte >> 4) & 0x03, (byte >> 2) & 0x03, byte & 0x03))
    return pixels[: width * height]


def decode_tiled8_4bpp(
    payload: bytes,
    width: int,
    height: int,
    nibble_order: NibbleOrder,
) -> list[int]:
    if width % 8 or height % 8:
        raise ValueError("tiled8 layout needs cell dimensions divisible by 8")

    pixels = [0] * (width * height)
    cursor = 0
    for tile_y in range(0, height, 8):
        for tile_x in range(0, width, 8):
            tile = payload[cursor : cursor + 32]
            cursor += 32
            values = decode_linear_4bpp(tile, 8, 8, nibble_order)
            for y in range(8):
                for x in range(8):
                    pixels[(tile_y + y) * width + tile_x + x] = values[y * 8 + x]
    return pixels


def palette_rgba(value: int) -> tuple[int, int, int, int]:
    if value < 4:
        palette_2bpp = [
            (0, 0, 0, 0),
            (178, 178, 178, 255),
            (92, 92, 92, 255),
            (0, 0, 0, 255),
        ]
        return palette_2bpp[value]

    palette = [
        (0, 0, 0, 0),
        (20, 24, 28, 255),
        (66, 56, 46, 255),
        (98, 84, 66, 255),
        (134, 117, 90, 255),
        (172, 151, 113, 255),
        (211, 191, 146, 255),
        (246, 232, 185, 255),
        (35, 45, 70, 255),
        (57, 76, 112, 255),
        (84, 112, 154, 255),
        (120, 153, 193, 255),
        (158, 191, 223, 255),
        (193, 220, 241, 255),
        (226, 241, 250, 255),
        (255, 255, 255, 255),
    ]
    return palette[value & 0x0F]


def glyph_image(
    payload: bytes,
    width: int,
    height: int,
    bpp: int,
    layout: Layout,
    nibble_order: NibbleOrder,
    scale: int,
) -> Image.Image:
    if bpp == 2:
        values = decode_linear_2bpp(payload, width, height)
    elif layout == "tiled8":
        values = decode_tiled8_4bpp(payload, width, height, nibble_order)
    else:
        values = decode_linear_4bpp(payload, width, height, nibble_order)

    image = Image.new("RGBA", (width, height))
    image.putdata([palette_rgba(value) for value in values])
    if scale == 1:
        return image
    return image.resize((width * scale, height * scale), Image.Resampling.NEAREST)


def checkerboard(size: tuple[int, int], block: int) -> Image.Image:
    image = Image.new("RGBA", size, (246, 246, 240, 255))
    draw = ImageDraw.Draw(image)
    alt = (224, 224, 216, 255)
    for y in range(0, size[1], block):
        for x in range(0, size[0], block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=alt)
    return image


def export_raw_atlas(
    source: Path,
    out_png: Path,
    *,
    cell_width: int = 13,
    cell_height: int = 13,
    bpp: int = 2,
    layout: Layout = "tiled8",
    nibble_order: NibbleOrder = "high",
    offset: int | None = None,
    columns: int = 32,
    scale: int = 3,
    pad: int = 1,
) -> ExportConfig:
    data = source.read_bytes()
    glyph_size = bytes_per_glyph(cell_width, cell_height, bpp)
    actual_offset = choose_auto_offset(len(data), glyph_size) if offset is None else offset
    if actual_offset < 0 or actual_offset >= len(data):
        raise ValueError(f"offset {actual_offset} is outside file of size {len(data)}")

    usable = len(data) - actual_offset
    glyph_count = usable // glyph_size
    if glyph_count <= 0:
        raise ValueError("no full glyphs can be decoded with the selected settings")

    rows = math.ceil(glyph_count / columns)
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    atlas_w = columns * tile_w + (columns + 1) * pad
    atlas_h = rows * tile_h + (rows + 1) * pad
    atlas = checkerboard((atlas_w, atlas_h), max(2, scale * 2))

    for index in range(glyph_count):
        start = actual_offset + index * glyph_size
        glyph = glyph_image(
            data[start : start + glyph_size],
            cell_width,
            cell_height,
            bpp,
            layout,
            nibble_order,
            scale,
        )
        x = pad + (index % columns) * (tile_w + pad)
        y = pad + (index // columns) * (tile_h + pad)
        atlas.alpha_composite(glyph, (x, y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_png)

    config = ExportConfig(
        source=str(source),
        out_png=str(out_png),
        mode=f"raw-{bpp}bpp",
        cell_width=cell_width,
        cell_height=cell_height,
        bpp=bpp,
        layout=layout,
        nibble_order=nibble_order,
        offset=actual_offset,
        glyph_count=glyph_count,
        columns=columns,
        scale=scale,
        pad=pad,
        bytes_per_glyph=glyph_size,
        file_size=len(data),
        markers_found=find_nitro_markers(data),
    )

    metadata_path = out_png.with_suffix(".json")
    metadata_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return config


def export_rtfn_atlas(
    source: Path,
    out_png: Path,
    *,
    columns: int = 32,
    scale: int = 4,
    pad: int = 1,
) -> ExportConfig:
    data = source.read_bytes()
    sections = parse_sections(data)
    tglp = next(((off, size) for off, magic, size in sections if magic == SECTION_TGLP), None)
    if tglp is None:
        raise ValueError(f"{source} has RTFN header but no PLGC/TGLP section")

    tglp_off, tglp_size = tglp
    cell_width, cell_height, cell_size, _baseline, _max_width, bpp, _rot = struct.unpack_from(
        "<BBHBBBB", data, tglp_off + 8
    )
    glyph_count = (tglp_size - 16) // cell_size
    rows = math.ceil(glyph_count / columns)
    tile_w = cell_width * scale
    tile_h = cell_height * scale
    atlas = checkerboard(
        (columns * tile_w + (columns + 1) * pad, rows * tile_h + (rows + 1) * pad),
        max(2, scale * 2),
    )

    for index in range(glyph_count):
        start = tglp_off + 16 + index * cell_size
        glyph = glyph_image(
            data[start : start + cell_size],
            cell_width,
            cell_height,
            bpp,
            "linear",
            "high",
            scale,
        )
        x = pad + (index % columns) * (tile_w + pad)
        y = pad + (index // columns) * (tile_h + pad)
        atlas.alpha_composite(glyph, (x, y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_png)
    config = ExportConfig(
        source=str(source),
        out_png=str(out_png),
        mode="rtfn",
        cell_width=cell_width,
        cell_height=cell_height,
        bpp=bpp,
        layout="linear",
        nibble_order="high",
        offset=tglp_off + 16,
        glyph_count=glyph_count,
        columns=columns,
        scale=scale,
        pad=pad,
        bytes_per_glyph=cell_size,
        file_size=len(data),
        markers_found=find_nitro_markers(data),
    )
    out_png.with_suffix(".json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return config


def export_font_atlas(
    source: Path,
    out_png: Path,
    *,
    cell_width: int | None = None,
    cell_height: int | None = None,
    bpp: int | None = None,
    layout: Layout = "linear",
    nibble_order: NibbleOrder = "high",
    offset: int | None = None,
    columns: int = 32,
    scale: int = 4,
    pad: int = 1,
) -> ExportConfig:
    data = source.read_bytes()
    if data.startswith(b"RTFN"):
        return export_rtfn_atlas(source, out_png, columns=columns, scale=scale, pad=pad)

    return export_raw_atlas(
        source,
        out_png,
        cell_width=cell_width or 13,
        cell_height=cell_height or 13,
        bpp=bpp or 2,
        layout=layout,
        nibble_order=nibble_order,
        offset=offset,
        columns=columns,
        scale=scale,
        pad=pad,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an NFTR/raw 4bpp font atlas to PNG.")
    parser.add_argument("source", type=Path, help="Path to the source .NFTR file.")
    parser.add_argument("--out", type=Path, default=Path("data/processed/a_atlas.png"))
    parser.add_argument("--cell-width", type=int, default=None)
    parser.add_argument("--cell-height", type=int, default=None)
    parser.add_argument("--bpp", type=int, choices=(2, 4), default=None)
    parser.add_argument("--layout", choices=("tiled8", "linear"), default="linear")
    parser.add_argument("--nibble-order", choices=("high", "low"), default="high")
    parser.add_argument("--offset", type=int, default=None, help="Byte offset. Default: auto.")
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--pad", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = export_font_atlas(
        args.source,
        args.out,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        bpp=args.bpp,
        layout=args.layout,
        nibble_order=args.nibble_order,
        offset=args.offset,
        columns=args.columns,
        scale=args.scale,
        pad=args.pad,
    )
    print(json.dumps(asdict(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
