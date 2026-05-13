from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

from font_machine_learn.paths import TARGET_CONTACT, TARGET_DIR, TARGET_METADATA

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


@dataclass(frozen=True)
class WidthMetrics:
    left: int
    glyph_width: int
    advance: int


@dataclass(frozen=True)
class GlyphRecord:
    index: int
    codes: list[int]
    chars: list[str]
    width: WidthMetrics
    histogram: dict[str, int]
    png: str


@dataclass(frozen=True)
class RTFNFont:
    source: str
    cell_width: int
    cell_height: int
    cell_size: int
    baseline: int
    max_width: int
    bpp: int
    glyph_count: int
    glyph_offset: int
    widths: list[WidthMetrics]
    index_to_codes: dict[int, list[int]]
    data: bytes


@dataclass(frozen=True)
class DatasetExport:
    source: str
    out_dir: str
    metadata_json: str
    contact_sheet: str
    glyph_count: int
    mapped_glyph_count: int
    code_count: int
    cell_width: int
    cell_height: int
    bpp: int


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


def decode_shift_jis_code(code: int) -> str | None:
    try:
        if code <= 0xFF:
            return bytes([code]).decode("shift_jis")
        return bytes([code >> 8, code & 0xFF]).decode("shift_jis")
    except UnicodeDecodeError:
        return None


def parse_cmap(data: bytes, sections: list[tuple[int, bytes, int]]) -> dict[int, list[int]]:
    index_to_codes: dict[int, list[int]] = {}
    for off, magic, _size in sections:
        if magic != SECTION_CMAP:
            continue

        first, last, cmap_type, _unknown, _next_body = struct.unpack_from("<HHHHI", data, off + 8)
        pos = off + 20
        if cmap_type == 0:
            first_index = struct.unpack_from("<H", data, pos)[0]
            for code in range(first, last + 1):
                index = first_index + code - first
                index_to_codes.setdefault(index, []).append(code)
        elif cmap_type == 1:
            for code in range(first, last + 1):
                index = struct.unpack_from("<H", data, pos)[0]
                pos += 2
                if index != 0xFFFF:
                    index_to_codes.setdefault(index, []).append(code)
        elif cmap_type == 2:
            count = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            for _ in range(count):
                code, index = struct.unpack_from("<HH", data, pos)
                pos += 4
                index_to_codes.setdefault(index, []).append(code)
        else:
            raise ValueError(f"unsupported PAMC/CMAP type {cmap_type} at 0x{off:X}")

    return {index: sorted(codes) for index, codes in sorted(index_to_codes.items())}


def parse_rtfn_font(source: Path) -> RTFNFont:
    data = source.read_bytes()
    if not data.startswith(b"RTFN"):
        raise ValueError(f"{source} is not a reversed-tag RTFN font")

    sections = parse_sections(data)
    tglp = next(((off, size) for off, magic, size in sections if magic == SECTION_TGLP), None)
    cwdh = next(((off, size) for off, magic, size in sections if magic == SECTION_CWDH), None)
    if tglp is None:
        raise ValueError(f"{source} has no PLGC glyph section")
    if cwdh is None:
        raise ValueError(f"{source} has no HDWC width section")

    tglp_off, tglp_size = tglp
    cell_width, cell_height, cell_size, baseline, max_width, bpp, _rot = struct.unpack_from(
        "<BBHBBBB", data, tglp_off + 8
    )
    glyph_count = (tglp_size - 16) // cell_size
    glyph_offset = tglp_off + 16

    cwdh_off, cwdh_size = cwdh
    first_index, last_index, _next_body = struct.unpack_from("<HHI", data, cwdh_off + 8)
    width_count = last_index - first_index + 1
    if width_count > glyph_count:
        raise ValueError(f"HDWC width count {width_count} exceeds glyph count {glyph_count}")

    widths = [WidthMetrics(0, max_width, max_width) for _ in range(glyph_count)]
    pos = cwdh_off + 16
    for index in range(first_index, last_index + 1):
        left, glyph_width, advance = struct.unpack_from("bbb", data, pos)
        pos += 3
        widths[index] = WidthMetrics(left, glyph_width, advance)

    return RTFNFont(
        source=str(source),
        cell_width=cell_width,
        cell_height=cell_height,
        cell_size=cell_size,
        baseline=baseline,
        max_width=max_width,
        bpp=bpp,
        glyph_count=glyph_count,
        glyph_offset=glyph_offset,
        widths=widths,
        index_to_codes=parse_cmap(data, sections),
        data=data,
    )


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


def decode_glyph_values(
    payload: bytes,
    width: int,
    height: int,
    bpp: int,
    layout: Layout = "linear",
    nibble_order: NibbleOrder = "high",
) -> list[int]:
    if bpp == 2:
        return decode_linear_2bpp(payload, width, height)
    if layout == "tiled8":
        return decode_tiled8_4bpp(payload, width, height, nibble_order)
    return decode_linear_4bpp(payload, width, height, nibble_order)


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
    values = decode_glyph_values(payload, width, height, bpp, layout, nibble_order)

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


def glyph_payload(font: RTFNFont, index: int) -> bytes:
    start = font.glyph_offset + index * font.cell_size
    return font.data[start : start + font.cell_size]


def glyph_histogram(values: list[int]) -> dict[str, int]:
    return {str(level): values.count(level) for level in range(max(values, default=0) + 1)}


def code_filename_part(codes: list[int]) -> str:
    if not codes:
        return "unmapped"
    return "sjis" + "-".join(f"{code:04X}" for code in codes[:3])


def export_target_dataset(
    source: Path,
    out_dir: Path = TARGET_DIR,
    *,
    contact_sheet: Path | None = TARGET_CONTACT,
    metadata_json: Path | None = TARGET_METADATA,
    scale: int = 4,
    columns: int = 32,
    pad: int = 1,
) -> DatasetExport:
    font = parse_rtfn_font(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_json or out_dir.parent / "target_metadata.json"
    contact_path = contact_sheet or out_dir.parent / "target_contact.png"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[GlyphRecord] = []
    tile_w = font.cell_width * scale
    tile_h = font.cell_height * scale
    rows = math.ceil(font.glyph_count / columns)
    sheet = checkerboard(
        (columns * tile_w + (columns + 1) * pad, rows * tile_h + (rows + 1) * pad),
        max(2, scale * 2),
    )

    for index in range(font.glyph_count):
        payload = glyph_payload(font, index)
        values = decode_glyph_values(payload, font.cell_width, font.cell_height, font.bpp)
        codes = font.index_to_codes.get(index, [])
        chars = [char for code in codes if (char := decode_shift_jis_code(code)) is not None]
        filename = f"glyph_{index:04d}_{code_filename_part(codes)}.png"
        glyph_path = out_dir / filename
        image = glyph_image(payload, font.cell_width, font.cell_height, font.bpp, "linear", "high", 1)
        image.save(glyph_path)

        scaled = image.resize((tile_w, tile_h), Image.Resampling.NEAREST)
        x = pad + (index % columns) * (tile_w + pad)
        y = pad + (index // columns) * (tile_h + pad)
        sheet.alpha_composite(scaled, (x, y))

        records.append(
            GlyphRecord(
                index=index,
                codes=codes,
                chars=chars,
                width=font.widths[index],
                histogram=glyph_histogram(values),
                png=str(glyph_path),
            )
        )

    sheet.save(contact_path)
    payload = {
        "source": str(source),
        "cell_width": font.cell_width,
        "cell_height": font.cell_height,
        "cell_size": font.cell_size,
        "baseline": font.baseline,
        "max_width": font.max_width,
        "bpp": font.bpp,
        "glyph_count": font.glyph_count,
        "mapped_glyph_count": sum(1 for record in records if record.codes),
        "code_count": sum(len(record.codes) for record in records),
        "glyph_dir": str(out_dir),
        "contact_sheet": str(contact_path),
        "glyphs": [asdict(record) for record in records],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return DatasetExport(
        source=str(source),
        out_dir=str(out_dir),
        metadata_json=str(metadata_path),
        contact_sheet=str(contact_path),
        glyph_count=font.glyph_count,
        mapped_glyph_count=payload["mapped_glyph_count"],
        code_count=payload["code_count"],
        cell_width=font.cell_width,
        cell_height=font.cell_height,
        bpp=font.bpp,
    )


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
