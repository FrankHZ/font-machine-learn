import argparse
import math
import re
import struct
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SECTION_FINF = b"FNIF"
SECTION_TGLP = b"PLGC"
SECTION_CWDH = b"HDWC"
SECTION_CMAP = b"PAMC"

GLYPH_SUBSTITUTIONS = {
    "…": "‥",
}

FORCE_REUSE_CHARS = set("一二三")
VERTICAL_CENTER_CJK = set("一二三")


def parse_key_values(line):
    values = {}
    for key, quoted, raw in re.findall(r'(\w+)=(?:"([^"]*)"|(-?\d+))', line):
        values[key] = quoted if quoted != "" else raw
    return values


def parse_fnt(path):
    info = {}
    common = {}
    pages = {}
    chars = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("info "):
                info = parse_key_values(line)
            elif line.startswith("common "):
                common = parse_key_values(line)
            elif line.startswith("page "):
                data = parse_key_values(line)
                pages[int(data["id"])] = data["file"]
            elif line.startswith("char id="):
                data = parse_key_values(line)
                char_id = int(data["id"])
                chars[char_id] = {
                    "id": char_id,
                    "x": int(data["x"]),
                    "y": int(data["y"]),
                    "width": int(data["width"]),
                    "height": int(data["height"]),
                    "xoffset": int(data["xoffset"]),
                    "yoffset": int(data["yoffset"]),
                    "xadvance": int(data["xadvance"]),
                    "page": int(data["page"]),
                }

    return info, common, pages, chars


def parse_nftr_source(path):
    data = path.read_bytes()
    sections = []
    off = 0x10
    while off + 8 <= len(data):
        magic = data[off : off + 4]
        size = struct.unpack_from("<I", data, off + 4)[0]
        sections.append((off, magic, size))
        off += size
        if off >= len(data):
            break

    tglp_off = next(o for o, magic, _size in sections if magic == SECTION_TGLP)
    tglp_size = next(size for o, magic, size in sections if magic == SECTION_TGLP)
    cwdh_off = next(o for o, magic, _size in sections if magic == SECTION_CWDH)
    cell_width, cell_height, cell_size, _baseline, _max_width, bpp, _rot = struct.unpack_from("<BBHBBBB", data, tglp_off + 8)
    if bpp != 2:
        raise ValueError(f"{path} is not a 2bpp NFTR")

    glyph_count = (tglp_size - 16) // cell_size
    code_to_index = {}
    for off, magic, size in sections:
        if magic != SECTION_CMAP:
            continue
        first, last, cmap_type, _unknown, _next_body = struct.unpack_from("<HHHHI", data, off + 8)
        pos = off + 20
        if cmap_type == 0:
            first_index = struct.unpack_from("<H", data, pos)[0]
            for code in range(first, last + 1):
                code_to_index[code] = first_index + code - first
        elif cmap_type == 1:
            for code in range(first, last + 1):
                index = struct.unpack_from("<H", data, pos)[0]
                pos += 2
                if index != 0xFFFF:
                    code_to_index[code] = index
        elif cmap_type == 2:
            count = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            for _ in range(count):
                code, index = struct.unpack_from("<HH", data, pos)
                pos += 4
                code_to_index[code] = index
        else:
            raise ValueError(f"Unsupported CMAP type {cmap_type} in {path}")

    code_to_glyph = {}
    char_to_glyph = {}
    for code, index in code_to_index.items():
        if index >= glyph_count:
            continue
        glyph_start = tglp_off + 16 + index * cell_size
        width_start = cwdh_off + 16 + index * 3
        glyph = {
            "blob": data[glyph_start : glyph_start + cell_size],
            "width": struct.unpack_from("bbb", data, width_start),
        }
        code_to_glyph[code] = glyph
        try:
            if code <= 0xFF:
                char = bytes([code]).decode("shift_jis")
            else:
                char = bytes([code >> 8, code & 0xFF]).decode("shift_jis")
            char_to_glyph.setdefault(char, glyph)
        except UnicodeDecodeError:
            pass

    return {
        "cell_width": cell_width,
        "cell_height": cell_height,
        "cell_size": cell_size,
        "code_to_glyph": code_to_glyph,
        "char_to_glyph": char_to_glyph,
    }


def load_mapping(path, fnt_chars, start_code):
    if path is None:
        entries = []
        for char_id in sorted(fnt_chars):
            if char_id <= 0xFFFF:
                entries.append((char_id, chr(char_id)))
        return entries

    entries = []
    next_code = start_code
    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line or line.lstrip().startswith("#"):
                continue
            if "=" in line:
                code_text, char_text = line.split("=", 1)
                code = int(code_text.strip(), 16)
                char = char_text[0]
            else:
                char = line[0]
                while next_code & 0xFF in (0x0A, 0x0D):
                    next_code += 1
                code = next_code
                next_code += 1
            entries.append((code, char))
    return entries


def load_pages(fnt_path, pages, required_page_ids):
    images = {}
    missing = []
    for page_id in sorted(required_page_ids):
        page_file = pages.get(page_id)
        if page_file is None:
            missing.append(f"page id {page_id} is not declared in the .fnt")
            continue
        image_path = fnt_path.parent / page_file
        if not image_path.exists():
            missing.append(str(image_path))
            continue
        images[page_id] = Image.open(image_path).convert("L")

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing BMFont page image(s):\n  {joined}")

    return images


def quantize_alpha(value, invert=False):
    level = int(round((value / 255.0) * 3.0))
    level = max(0, min(3, level))
    return 3 - level if invert else level


def pack_2bpp(values):
    out = bytearray(math.ceil(len(values) * 2 / 8))
    bit_pos = 0
    for value in values:
        byte_index = bit_pos // 8
        shift = 6 - (bit_pos % 8)
        out[byte_index] |= (value & 0x03) << shift
        bit_pos += 2
    return bytes(out)


def render_glyph(char_info, pages, cell_width, cell_height, invert=False):
    page = pages[char_info["page"]]
    crop = page.crop(
        (
            char_info["x"],
            char_info["y"],
            char_info["x"] + char_info["width"],
            char_info["y"] + char_info["height"],
        )
    )

    values = []
    for y in range(cell_height):
        for x in range(cell_width):
            src_x = x - char_info["xoffset"]
            src_y = y - char_info["yoffset"]
            if 0 <= src_x < crop.width and 0 <= src_y < crop.height:
                values.append(quantize_alpha(crop.getpixel((src_x, src_y)), invert))
            else:
                values.append(0)
    return pack_2bpp(values)


def glyph_placement(char, bbox, cell_width, cell_height, x_offset=0, y_offset=0):
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (cell_width - text_width) // 2 - bbox[0] + x_offset
    y = (cell_height - text_height) // 2 - bbox[1] + y_offset

    if char == "，":
        x = -bbox[0]
        y = cell_height - text_height - bbox[1]
    elif char == "；":
        x = -bbox[0]

    return x, y


def render_ttf_mask(font, char, cell_width, cell_height, x_offset=0, y_offset=0):
    canvas = Image.new("L", (cell_width * 3, cell_height * 3), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), char, font=font)
    x, y = glyph_placement(char, bbox, cell_width, cell_height, x_offset, y_offset)
    draw.text((x, y), char, fill=255, font=font)
    return canvas.crop((0, 0, cell_width, cell_height)), bbox


def levels_bbox(levels):
    points = [(x, y) for y, row in enumerate(levels) for x, value in enumerate(row) if value]
    if not points:
        return 0, 0, 0, 0
    left = min(x for x, _y in points)
    top = min(y for _x, y in points)
    right = max(x for x, _y in points) + 1
    bottom = max(y for _x, y in points) + 1
    return left, top, right, bottom


def align_levels(levels, char, cell_width, cell_height):
    left, top, right, bottom = levels_bbox(levels)
    if right <= left or bottom <= top:
        return levels

    align_left = is_cjk_char(char) or char in "，；"
    align_bottom = (is_cjk_char(char) and char not in VERTICAL_CENTER_CJK) or char == "，"
    dx = -left if align_left else 0
    dy = cell_height - bottom if align_bottom else 0
    if dx == 0 and dy == 0:
        return levels

    moved = [[0 for _x in range(cell_width)] for _y in range(cell_height)]
    for y in range(cell_height):
        for x in range(cell_width):
            value = levels[y][x]
            if not value:
                continue
            tx = x + dx
            ty = y + dy
            if 0 <= tx < cell_width and 0 <= ty < cell_height:
                moved[ty][tx] = max(moved[ty][tx], value)
    return moved


def patch_shadow_holes(levels, cell_width, cell_height):
    patched = [row[:] for row in levels]
    for y in range(cell_height):
        for x in range(cell_width):
            if levels[y][x] != 0:
                continue
            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < cell_width and 0 <= ny < cell_height:
                        neighbors.append(levels[ny][nx])
            if 1 in neighbors and 3 in neighbors and sum(1 for value in neighbors if value) >= 3:
                patched[y][x] = 1
    return patched


def remove_cjk_speckles(levels, cell_width, cell_height):
    cleaned = [row[:] for row in levels]
    for y in range(cell_height):
        for x in range(cell_width):
            if levels[y][x] != 3:
                continue
            main_neighbors = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < cell_width and 0 <= ny < cell_height and levels[ny][nx] == 3:
                        main_neighbors += 1
            if main_neighbors == 0:
                cleaned[y][x] = 0
    return cleaned


def render_ttf_shadow_glyph(
    font,
    char,
    cell_width,
    cell_height,
    threshold=96,
    x_offset=0,
    y_offset=0,
    diagonal_only=False,
    patch_holes=False,
):
    mask, bbox = render_ttf_mask(font, char, cell_width, cell_height, x_offset, y_offset)
    quantized_main = not diagonal_only and patch_holes
    solid = [[mask.getpixel((x, y)) >= threshold for x in range(cell_width)] for y in range(cell_height)]
    levels = [[0 for _x in range(cell_width)] for _y in range(cell_height)]

    for y in range(cell_height):
        for x in range(cell_width):
            if solid[y][x]:
                value = mask.getpixel((x, y))
                levels[y][x] = 2 if quantized_main and value < 192 else 3

    if is_cjk_char(char):
        levels = remove_cjk_speckles(levels, cell_width, cell_height)

    for y in range(cell_height):
        for x in range(cell_width):
            if not solid[y][x]:
                continue
            shadow_pixels = ((1, 1, 1),) if diagonal_only or quantized_main else ((1, 1, 1), (1, 0, 2), (0, 1, 2))
            for dx, dy, value in shadow_pixels:
                sx = x + dx
                sy = y + dy
                if 0 <= sx < cell_width and 0 <= sy < cell_height and levels[sy][sx] == 0:
                    levels[sy][sx] = value

    levels = align_levels(levels, char, cell_width, cell_height)
    if patch_holes:
        levels = patch_shadow_holes(levels, cell_width, cell_height)

    return pack_2bpp([value for row in levels for value in row]), bbox, levels_bbox(levels)


def is_cjk_char(char):
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


def is_latin_digit_or_punct(char):
    code = ord(char)
    category = unicodedata.category(char)
    return code < 0x80 or category[0] in "PS" or 0xFF00 <= code <= 0xFFEF or code == 0x3000


def build_runs(codes):
    runs = []
    if not codes:
        return runs

    start = prev = codes[0]
    for code in codes[1:]:
        if code == prev + 1:
            prev = code
            continue
        runs.append((start, prev))
        start = prev = code
    runs.append((start, prev))
    return runs


def make_header(file_size, section_count):
    return b"RTFN" + struct.pack("<HHIHH", 0xFEFF, 0x0101, file_size, 0x10, section_count)


def make_finf(line_feed, default_width, encoding, tglp_body, cwdh_body, cmap_body):
    body = struct.pack(
        "<BBHbbbBIII",
        0,
        line_feed,
        0,
        0,
        default_width,
        default_width,
        encoding,
        tglp_body,
        cwdh_body,
        cmap_body,
    )
    return SECTION_FINF + struct.pack("<I", 8 + len(body)) + body


def make_tglp(glyph_blobs, cell_width, cell_height, baseline, max_width):
    cell_size = math.ceil(cell_width * cell_height * 2 / 8)
    body = struct.pack("<BBHBBBB", cell_width, cell_height, cell_size, baseline, max_width, 2, 0)
    body += b"".join(glyph_blobs)
    return SECTION_TGLP + struct.pack("<I", 8 + len(body)) + body


def make_cwdh(widths):
    body = struct.pack("<HHI", 0, len(widths) - 1, 0)
    for left, glyph_width, advance in widths:
        body += struct.pack("<bbb", left, glyph_width, advance)
    while len(body) % 4:
        body += b"\x00"
    return SECTION_CWDH + struct.pack("<I", 8 + len(body)) + body


def make_cmaps(code_to_index, cmap_body_offsets):
    chunks = []
    codes = sorted(code_to_index)
    runs = build_runs(codes)

    for i, (start, end) in enumerate(runs):
        first_index = code_to_index[start]
        expected = list(range(first_index, first_index + end - start + 1))
        actual = [code_to_index[code] for code in range(start, end + 1)]
        if actual != expected:
            raise ValueError("Internal CMAP grouping error: glyph indexes are not contiguous")

        next_body = cmap_body_offsets[i + 1] if i + 1 < len(cmap_body_offsets) else 0
        body = struct.pack("<HHHHIH", start, end, 0, 0, next_body, first_index)
        body += b"\x00\x00"
        chunks.append(SECTION_CMAP + struct.pack("<I", 8 + len(body)) + body)

    return chunks


def bounded_byte(value, label):
    if not -128 <= value <= 127:
        raise ValueError(f"{label}={value} does not fit in signed NFTR width byte")
    return value


def resolve_existing_path(path, base_dir):
    if path.is_absolute():
        return path.resolve()
    base_path = base_dir / path
    if base_path.exists():
        return base_path.resolve()
    cwd_path = Path.cwd() / path
    return cwd_path.resolve()


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Build a 2bpp Nintendo DS NFTR from a BMFont .fnt and page PNGs.")
    parser.add_argument("--fnt", default="../../fonts/full-semibold-18.fnt", type=Path)
    parser.add_argument("--ttf", type=Path, help="Render glyphs directly from a TrueType/OpenType/TTC font.")
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=13)
    parser.add_argument("--render-mode", choices=("quantize", "shadow", "shadow-diagonal", "shadow-quantized"), default="quantize")
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--x-offset", type=int, default=0)
    parser.add_argument("--y-offset", type=int, default=0)
    parser.add_argument("--patch-shadow-holes", action="store_true")
    parser.add_argument("--reuse-nftr", type=Path, help="Reuse selected glyph bitmaps and CWDH widths from an existing NFTR.")
    parser.add_argument("--reuse-kind", choices=("latin-punct", "all-present"), default="latin-punct")
    parser.add_argument("--cjk-advance", type=int, default=None)
    parser.add_argument("--cjk-left", type=int, default=0)
    parser.add_argument("--map", default="a.txt", type=Path, help="Optional CT2-style CODE=char map. Omit for Unicode codes.")
    parser.add_argument("--out", default="font-2bpp.nftr", type=Path)
    parser.add_argument("--cell-width", type=int, default=None)
    parser.add_argument("--cell-height", type=int, default=None)
    parser.add_argument("--baseline", type=int, default=None)
    parser.add_argument("--encoding", type=int, default=2, help="NFTR encoding byte. test.nftr uses 2.")
    parser.add_argument("--start-code", type=lambda s: int(s, 0), default=0xE800)
    parser.add_argument("--invert", action="store_true", help="Invert quantized 2bpp levels.")
    parser.add_argument("--missing", choices=("error", "blank"), default="error")
    args = parser.parse_args()

    map_path = args.map if args.map.is_absolute() else script_dir / args.map
    out_path = args.out if args.out.is_absolute() else script_dir / args.out
    reuse_source = None
    if args.reuse_nftr:
        reuse_path = resolve_existing_path(args.reuse_nftr, script_dir)
        reuse_source = parse_nftr_source(reuse_path)

    font = None
    if args.ttf:
        font_path = resolve_existing_path(args.ttf, script_dir)
        font = ImageFont.truetype(str(font_path), size=args.font_size, index=args.font_index)
        info = {}
        common = {"lineHeight": str(args.cell_height or 15), "base": str(args.baseline or 15)}
        pages = {}
        fnt_chars = {}
    else:
        fnt_path = resolve_existing_path(args.fnt, script_dir)
        info, common, pages, fnt_chars = parse_fnt(fnt_path)

    entries = load_mapping(map_path if map_path.exists() else None, fnt_chars, args.start_code)

    glyph_specs = []
    skipped = []
    required_pages = set()
    for code, char in entries:
        char_id = ord(char)
        if font is not None:
            glyph_specs.append((code, char, {"id": char_id}))
        else:
            char_info = fnt_chars.get(char_id)
            if char_info is None:
                if args.missing == "error":
                    skipped.append(f"U+{char_id:04X} {char!r}")
                    continue
                glyph_specs.append((code, char, None))
            else:
                required_pages.add(char_info["page"])
                glyph_specs.append((code, char, char_info))

    if skipped:
        joined = "\n  ".join(skipped[:50])
        extra = "" if len(skipped) <= 50 else f"\n  ... and {len(skipped) - 50} more"
        raise KeyError(f"Character(s) missing from BMFont:\n  {joined}{extra}")

    page_images = {} if font is not None else load_pages(fnt_path, pages, required_pages)

    if font is not None:
        max_source_width = 15
        max_source_height = 15
    else:
        max_source_width = max((spec["width"] + max(0, spec["xoffset"]) for _, _, spec in glyph_specs if spec), default=1)
        max_source_height = max((spec["height"] + max(0, spec["yoffset"]) for _, _, spec in glyph_specs if spec), default=1)
    cell_width = args.cell_width or min(127, max_source_width)
    cell_height = args.cell_height or int(common.get("lineHeight", max_source_height))
    baseline = args.baseline if args.baseline is not None else int(common.get("base", cell_height))
    default_width = min(127, cell_width)

    cell_size = math.ceil(cell_width * cell_height * 2 / 8)
    blank = b"\x00" * cell_size
    glyph_blobs = []
    widths = []
    code_to_index = {}
    reused_count = 0
    cjk_adjusted_count = 0

    for index, (code, char, char_info) in enumerate(glyph_specs):
        code_to_index[code] = index
        if reuse_source is not None:
            reusable = args.reuse_kind == "all-present" or is_latin_digit_or_punct(char) or char in FORCE_REUSE_CHARS
            source_glyph = reuse_source["code_to_glyph"].get(code)
            if source_glyph is None:
                source_glyph = reuse_source["char_to_glyph"].get(char)
            if source_glyph is None:
                source_glyph = reuse_source["char_to_glyph"].get(GLYPH_SUBSTITUTIONS.get(char, ""))
            if reusable and source_glyph is not None:
                if reuse_source["cell_width"] != cell_width or reuse_source["cell_height"] != cell_height:
                    raise ValueError("Reused NFTR cell size does not match output cell size")
                glyph_blobs.append(source_glyph["blob"])
                widths.append(source_glyph["width"])
                reused_count += 1
                continue

        if char_info is None:
            glyph_blobs.append(blank)
            widths.append((0, 0, default_width))
            continue

        if font is not None:
            if args.render_mode in ("shadow", "shadow-diagonal", "shadow-quantized"):
                blob, bbox, pixel_bbox = render_ttf_shadow_glyph(
                    font,
                    char,
                    cell_width,
                    cell_height,
                    args.threshold,
                    args.x_offset,
                    args.y_offset,
                    args.render_mode == "shadow-diagonal",
                    args.patch_shadow_holes or args.render_mode == "shadow-quantized",
                )
            else:
                mask, bbox = render_ttf_mask(font, char, cell_width, cell_height, args.x_offset, args.y_offset)
                blob = pack_2bpp([quantize_alpha(mask.getpixel((x, y)), args.invert) for y in range(cell_height) for x in range(cell_width)])
                pixel_bbox = mask.getbbox() or (0, 0, 0, 0)
            glyph_blobs.append(blob)
            bbox_width = max(1, pixel_bbox[2] - pixel_bbox[0])
            left = args.cjk_left if is_cjk_char(char) and args.cjk_advance is not None else 0
            glyph_width = min(cell_width, bbox_width)
            if is_cjk_char(char) and args.cjk_advance is not None:
                advance = args.cjk_advance
                cjk_adjusted_count += 1
            elif is_cjk_char(char):
                advance = glyph_width
                cjk_adjusted_count += 1
            else:
                advance = glyph_width
        else:
            glyph_blobs.append(render_glyph(char_info, page_images, cell_width, cell_height, args.invert))
            left = char_info["xoffset"]
            glyph_width = min(cell_width, char_info["width"])
            advance = min(127, char_info["xadvance"])
        left = bounded_byte(left, "xoffset")
        glyph_width = bounded_byte(glyph_width, "glyph width")
        advance = bounded_byte(advance, "xadvance")
        widths.append((left, glyph_width, advance))

    tglp = make_tglp(glyph_blobs, cell_width, cell_height, baseline, default_width)
    cwdh = make_cwdh(widths)

    cmap_runs = build_runs(sorted(code_to_index))
    cmaps_start = 0x10 + 0x1C + len(tglp) + len(cwdh)
    cmap_body_offsets = [cmaps_start + i * 24 + 8 for i in range(len(cmap_runs))]
    cmaps = make_cmaps(code_to_index, cmap_body_offsets)

    finf_len = 0x1C
    tglp_body = 0x10 + finf_len + 8
    cwdh_body = 0x10 + finf_len + len(tglp) + 8
    cmap_body = cmap_body_offsets[0] if cmap_body_offsets else 0
    finf = make_finf(min(255, int(common.get("lineHeight", cell_height))), default_width, args.encoding, tglp_body, cwdh_body, cmap_body)

    sections = [finf, tglp, cwdh] + cmaps
    file_size = 0x10 + sum(len(section) for section in sections)
    out = make_header(file_size, len(sections)) + b"".join(sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)

    print(f"Wrote {out_path}")
    print(f"glyphs: {len(glyph_blobs)}")
    print(f"cell: {cell_width}x{cell_height}, {cell_size} bytes/glyph, 2bpp")
    print(f"CMAP chunks: {len(cmaps)}")
    if reuse_source is not None:
        print(f"reused glyphs: {reused_count}")
    if args.cjk_advance is not None:
        print(f"CJK adjusted widths: {cjk_adjusted_count}, advance={args.cjk_advance}, left={args.cjk_left}")
    print(f"file size: {file_size} bytes")


if __name__ == "__main__":
    main()
