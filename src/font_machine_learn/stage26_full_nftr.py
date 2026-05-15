from __future__ import annotations

import json
import math
import struct
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sklearn.exceptions import ConvergenceWarning

from font_machine_learn.baseline import image_to_target_levels, levels_to_image
from font_machine_learn.nftr import (
    SECTION_CMAP,
    SECTION_CWDH,
    SECTION_FINF,
    SECTION_TGLP,
    WidthMetrics,
    decode_shift_jis_code,
    export_font_atlas,
    glyph_payload,
    parse_rtfn_font,
)
from font_machine_learn.paths import SONG13_LAYER_MLP_NFTR_DIR, TARGET_METADATA
from font_machine_learn.song13_layer_mlp import (
    build_target_ge2_training_rows,
    predict_source_locked_levels,
    train_binary_models,
)
from font_machine_learn.stage26_nftr import pack_2bpp_values


GLYPH_SUBSTITUTIONS = {"…": "‥"}
# These tiny horizontal-stroke glyphs are very sensitive to vertical hinting.
# The original NFTR versions match the target style better than rerendered Song13.
SIMPLE_STROKE_REUSE_CHARS = set("一二三")
VERTICAL_CENTER_CJK = set("一二三")
PADDED_PUNCTUATION_ADVANCE = {"，": 6, "；": 6}


@dataclass(frozen=True)
class FullStage26NFTRExport:
    source_nftr: str
    map_file: str
    target_metadata: str
    out_nftr: str
    metadata_json: str
    preview_png: str
    glyph_count: int
    reused_glyph_count: int
    rendered_glyph_count: int
    cmap_count: int
    cell_width: int
    cell_height: int
    bpp: int
    bytes_per_glyph: int
    start_code: int
    punctuation_padded_advance: dict[str, int]


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


def is_latin_digit_or_punct(char: str) -> bool:
    code = ord(char)
    category = unicodedata.category(char)
    return code < 0x80 or category[0] in "PS" or 0xFF00 <= code <= 0xFFEF or code == 0x3000


def load_char_sequence(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8-sig"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8")
    chars = [char for char in text if char not in "\r\n"]
    seen: set[str] = set()
    duplicates = [char for char in chars if char in seen or seen.add(char)]
    if duplicates:
        raise ValueError(f"{path} contains duplicate characters, first duplicate: {duplicates[0]!r}")
    return chars


def load_mapping_entries(path: Path, start_code: int) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if lines and all("=" in line for line in lines):
        entries: list[tuple[int, str]] = []
        seen_codes: set[int] = set()
        seen_chars: set[str] = set()
        for line in lines:
            code_text, char_text = line.split("=", 1)
            code = int(code_text.strip(), 16)
            char = char_text[0]
            if code in seen_codes:
                raise ValueError(f"{path} contains duplicate code: {code_text}")
            if char in seen_chars:
                raise ValueError(f"{path} contains duplicate character: {char!r}")
            seen_codes.add(code)
            seen_chars.add(char)
            entries.append((code, char))
        return entries
    return assign_codes(load_char_sequence(path), start_code)


def assign_codes(chars: list[str], start_code: int) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    code = start_code
    for char in chars:
        while code & 0xFF in (0x0A, 0x0D):
            code += 1
        entries.append((code, char))
        code += 1
    return entries


def build_runs(codes: list[int]) -> list[tuple[int, int]]:
    if not codes:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = codes[0]
    for code in codes[1:]:
        if code == prev + 1:
            prev = code
            continue
        runs.append((start, prev))
        start = prev = code
    runs.append((start, prev))
    return runs


def make_header(file_size: int, section_count: int) -> bytes:
    return b"RTFN" + struct.pack("<HHIHH", 0xFEFF, 0x0101, file_size, 0x10, section_count)


def make_finf(line_feed: int, default_width: int, encoding: int, tglp_body: int, cwdh_body: int, cmap_body: int) -> bytes:
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


def make_tglp(glyph_blobs: list[bytes], cell_width: int, cell_height: int, baseline: int, max_width: int) -> bytes:
    cell_size = math.ceil(cell_width * cell_height * 2 / 8)
    body = struct.pack("<BBHBBBB", cell_width, cell_height, cell_size, baseline, max_width, 2, 0)
    body += b"".join(glyph_blobs)
    return SECTION_TGLP + struct.pack("<I", 8 + len(body)) + body


def make_cwdh(widths: list[WidthMetrics]) -> bytes:
    body = struct.pack("<HHI", 0, len(widths) - 1, 0)
    for width in widths:
        body += struct.pack("<bbb", width.left, width.glyph_width, width.advance)
    while len(body) % 4:
        body += b"\x00"
    return SECTION_CWDH + struct.pack("<I", 8 + len(body)) + body


def make_cmaps(code_to_index: dict[int, int], cmap_body_offsets: list[int]) -> list[bytes]:
    chunks: list[bytes] = []
    for i, (start, end) in enumerate(build_runs(sorted(code_to_index))):
        first_index = code_to_index[start]
        actual = [code_to_index[code] for code in range(start, end + 1)]
        expected = list(range(first_index, first_index + end - start + 1))
        if actual != expected:
            raise ValueError("CMAP run has non-contiguous glyph indexes")
        next_body = cmap_body_offsets[i + 1] if i + 1 < len(cmap_body_offsets) else 0
        body = struct.pack("<HHHHIH", start, end, 0, 0, next_body, first_index)
        body += b"\x00\x00"
        chunks.append(SECTION_CMAP + struct.pack("<I", 8 + len(body)) + body)
    return chunks


def levels_bbox(levels: list[list[int]]) -> tuple[int, int, int, int]:
    points = [(x, y) for y, row in enumerate(levels) for x, value in enumerate(row) if value]
    if not points:
        return 0, 0, 0, 0
    return (
        min(x for x, _y in points),
        min(y for _x, y in points),
        max(x for x, _y in points) + 1,
        max(y for _x, y in points) + 1,
    )


def render_mask(
    font: ImageFont.FreeTypeFont,
    char: str,
    *,
    cell_width: int,
    cell_height: int,
    font_mode: str,
    threshold: int,
    x_offset: int,
    y_offset: int,
) -> list[list[bool]]:
    canvas = Image.new("L", (cell_width * 3, cell_height * 3), 0)
    draw = ImageDraw.Draw(canvas)
    draw.fontmode = font_mode
    bbox = draw.textbbox((0, 0), char, font=font)
    if bbox is None:
        return [[False for _x in range(cell_width)] for _y in range(cell_height)]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (cell_width - width) // 2 - bbox[0] + x_offset
    y = (cell_height - height) // 2 - bbox[1] + y_offset
    if char == "，":
        x = -bbox[0]
        y = cell_height - height - bbox[1]
    elif char == "；":
        x = -bbox[0] + x_offset
    draw.text((x, y), char, fill=255, font=font)
    crop = canvas.crop((0, 0, cell_width, cell_height))
    mask = [[crop.getpixel((x, y)) >= threshold for x in range(cell_width)] for y in range(cell_height)]
    return align_mask(mask, char)


def align_mask(mask: list[list[bool]], char: str) -> list[list[bool]]:
    levels = [[3 if value else 0 for value in row] for row in mask]
    left, _top, right, bottom = levels_bbox(levels)
    if right <= left:
        return mask
    height = len(mask)
    width = len(mask[0]) if height else 0
    align_left = is_cjk_char(char) or char in "，；"
    align_bottom = (is_cjk_char(char) and char not in VERTICAL_CENTER_CJK) or char == "，"
    dx = -left if align_left else 0
    dy = height - bottom if align_bottom else 0
    if dx == 0 and dy == 0:
        return mask
    moved = [[False for _x in range(width)] for _y in range(height)]
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if not value:
                continue
            tx = x + dx
            ty = y + dy
            if 0 <= tx < width and 0 <= ty < height:
                moved[ty][tx] = True
    return moved


def original_glyph_maps(source_nftr: Path) -> tuple[dict[str, tuple[bytes, WidthMetrics]], dict[int, tuple[bytes, WidthMetrics]]]:
    font = parse_rtfn_font(source_nftr)
    char_map: dict[str, tuple[bytes, WidthMetrics]] = {}
    code_map: dict[int, tuple[bytes, WidthMetrics]] = {}
    for index in range(font.glyph_count):
        payload = glyph_payload(font, index)
        width = font.widths[index]
        for code in font.index_to_codes.get(index, []):
            code_map[code] = (payload, width)
            char = decode_shift_jis_code(code)
            if char is not None:
                char_map.setdefault(char, (payload, width))
    return char_map, code_map


def reusable_original(
    char: str,
    code: int,
    *,
    char_map: dict[str, tuple[bytes, WidthMetrics]],
    code_map: dict[int, tuple[bytes, WidthMetrics]],
) -> tuple[bytes, WidthMetrics] | None:
    if char in PADDED_PUNCTUATION_ADVANCE:
        return None
    reusable = is_latin_digit_or_punct(char) or char in SIMPLE_STROKE_REUSE_CHARS
    if not reusable:
        return None
    return code_map.get(code) or char_map.get(char) or char_map.get(GLYPH_SUBSTITUTIONS.get(char, ""))


def width_from_levels(char: str, levels: list[list[int]], cell_width: int) -> WidthMetrics:
    left, _top, right, _bottom = levels_bbox(levels)
    glyph_width = max(1, min(cell_width, right - left))
    if char in PADDED_PUNCTUATION_ADVANCE:
        advance = min(cell_width, max(PADDED_PUNCTUATION_ADVANCE[char], glyph_width + 3))
        return WidthMetrics(0, advance, advance)
    return WidthMetrics(0, glyph_width, glyph_width)


def train_stage26_heads(
    target_metadata: Path,
    *,
    patch_radius: int,
    max_train_glyphs: int | None,
    core_hidden_units: int,
    shadow_hidden_units: int,
    max_iter: int,
    random_seed: int,
) -> tuple[object, object]:
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    x_core, y_core, x_shadow, y_shadow, _train_cjk = build_target_ge2_training_rows(
        list(target["glyphs"]),
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
    return core_models["core_patch_mlp"], shadow_models["shadow_logistic_balanced"]


def export_full_stage26_nftr(
    *,
    source_nftr: Path = Path("a.NFTR"),
    map_file: Path = Path("ds_nftr/a.txt"),
    target_metadata: Path = TARGET_METADATA,
    font_path: Path = Path("fonts/WenQuanYi.Bitmap.Song.13px.ttf"),
    font_index: int = 0,
    font_size: int = 15,
    out_nftr: Path | None = None,
    metadata_json: Path | None = None,
    preview_png: Path | None = None,
    start_code: int = 0xE800,
    cell_width: int = 15,
    cell_height: int = 15,
    baseline: int = 15,
    font_mode: str = "L",
    threshold: int = 96,
    x_offset: int = -1,
    y_offset: int = 1,
    patch_radius: int = 4,
    core_threshold: float = 0.55,
    shadow_threshold: float = 0.45,
    max_train_glyphs: int | None = None,
    core_hidden_units: int = 64,
    shadow_hidden_units: int = 64,
    max_iter: int = 80,
    random_seed: int = 26,
    encoding: int = 2,
    preview_columns: int = 32,
    preview_scale: int = 4,
) -> FullStage26NFTRExport:
    out_dir = out_nftr.parent if out_nftr else SONG13_LAYER_MLP_NFTR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    nftr_path = out_nftr or out_dir / "a-stage26-fullmap.NFTR"
    metadata_path = metadata_json or out_dir / "a-stage26-fullmap.json"
    preview_path = preview_png or out_dir / "a-stage26-fullmap-preview.png"

    entries = load_mapping_entries(map_file, start_code)
    char_map, code_map = original_glyph_maps(source_nftr)
    font = ImageFont.truetype(str(font_path), size=font_size, index=font_index)
    core_model, shadow_model = train_stage26_heads(
        target_metadata,
        patch_radius=patch_radius,
        max_train_glyphs=max_train_glyphs,
        core_hidden_units=core_hidden_units,
        shadow_hidden_units=shadow_hidden_units,
        max_iter=max_iter,
        random_seed=random_seed,
    )

    glyph_blobs: list[bytes] = []
    widths: list[WidthMetrics] = []
    code_to_index: dict[int, int] = {}
    rendered_count = 0
    reused_count = 0
    predicted_dir = out_dir / "fullmap_predicted_2bpp"
    predicted_dir.mkdir(parents=True, exist_ok=True)

    for index, (code, char) in enumerate(entries):
        code_to_index[code] = index
        original = reusable_original(char, code, char_map=char_map, code_map=code_map)
        if original is not None:
            blob, width = original
            reused_count += 1
        else:
            source_mask = render_mask(
                font,
                char,
                cell_width=cell_width,
                cell_height=cell_height,
                font_mode=font_mode,
                threshold=threshold,
                x_offset=x_offset,
                y_offset=y_offset,
            )
            levels = predict_source_locked_levels(
                core_model,
                shadow_model,
                source_mask,
                patch_radius=patch_radius,
                core_threshold=core_threshold,
                shadow_threshold=shadow_threshold,
            )
            blob = pack_2bpp_values([value for row in levels for value in row])
            width = width_from_levels(char, levels, cell_width)
            levels_to_image(levels).save(predicted_dir / f"glyph_{index:04d}_u{ord(char):04X}.png")
            rendered_count += 1
        glyph_blobs.append(blob)
        widths.append(width)

    tglp = make_tglp(glyph_blobs, cell_width, cell_height, baseline, cell_width)
    cwdh = make_cwdh(widths)
    cmap_runs = build_runs(sorted(code_to_index))
    cmaps_start = 0x10 + 0x1C + len(tglp) + len(cwdh)
    cmap_body_offsets = [cmaps_start + i * 24 + 8 for i in range(len(cmap_runs))]
    cmaps = make_cmaps(code_to_index, cmap_body_offsets)
    finf_len = 0x1C
    tglp_body = 0x10 + finf_len + 8
    cwdh_body = 0x10 + finf_len + len(tglp) + 8
    cmap_body = cmap_body_offsets[0] if cmap_body_offsets else 0
    finf = make_finf(cell_height, cell_width, encoding, tglp_body, cwdh_body, cmap_body)
    sections = [finf, tglp, cwdh] + cmaps
    out = make_header(0x10 + sum(len(section) for section in sections), len(sections)) + b"".join(sections)
    nftr_path.write_bytes(out)
    export_font_atlas(nftr_path, preview_path, columns=preview_columns, scale=preview_scale)

    result = FullStage26NFTRExport(
        source_nftr=str(source_nftr),
        map_file=str(map_file),
        target_metadata=str(target_metadata),
        out_nftr=str(nftr_path),
        metadata_json=str(metadata_path),
        preview_png=str(preview_path),
        glyph_count=len(glyph_blobs),
        reused_glyph_count=reused_count,
        rendered_glyph_count=rendered_count,
        cmap_count=len(cmaps),
        cell_width=cell_width,
        cell_height=cell_height,
        bpp=2,
        bytes_per_glyph=math.ceil(cell_width * cell_height * 2 / 8),
        start_code=start_code,
        punctuation_padded_advance=PADDED_PUNCTUATION_ADVANCE,
    )
    payload = asdict(result) | {
        "stage26_candidate": "core_patch_mlp_shadow_logistic_balanced_c055_s045",
        "stage26_thresholds": {"core": core_threshold, "shadow": shadow_threshold},
        "font": {
            "path": str(font_path),
            "index": font_index,
            "size": font_size,
            "mode": font_mode,
            "threshold": threshold,
            "x_offset": x_offset,
            "y_offset": y_offset,
        },
        "rules": [
            "Read CODE=char mappings from ds_nftr/a.txt when available; a plain char list falls back to sequential 0xE800 codes.",
            "Reuse original NFTR bitmap and CWDH for Latin/digits/punctuation when present.",
            "Reuse original NFTR glyphs for 一二三 because rerendered tiny horizontal strokes are hinting-sensitive.",
            "Substitute … with original ‥ when reusable.",
            "Render new CJK from Song13 and align visible pixels to left-bottom.",
            "Give ， and ； padded advance so punctuation does not crowd following text without taking a full cell.",
        ],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
