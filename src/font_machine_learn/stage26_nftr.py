from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from font_machine_learn.baseline import image_to_target_levels
from font_machine_learn.nftr import export_font_atlas, parse_rtfn_font
from font_machine_learn.paths import SONG13_LAYER_MLP_METADATA, SONG13_LAYER_MLP_NFTR_DIR


@dataclass(frozen=True)
class Stage26NFTRExport:
    source_nftr: str
    stage26_metadata: str
    candidate: str
    out_nftr: str
    metadata_json: str
    preview_png: str
    glyph_count: int
    cell_width: int
    cell_height: int
    bpp: int
    bytes_per_glyph: int
    patched_glyph_count: int


def pack_2bpp_values(values: list[int]) -> bytes:
    out = bytearray(math.ceil(len(values) * 2 / 8))
    bit_pos = 0
    for value in values:
        if value < 0 or value > 3:
            raise ValueError(f"2bpp pixel value must be in 0..3, got {value}")
        byte_index = bit_pos // 8
        shift = 6 - (bit_pos % 8)
        out[byte_index] |= value << shift
        bit_pos += 2
    return bytes(out)


def pack_levels_2bpp(levels: list[list[int]], width: int, height: int) -> bytes:
    if len(levels) != height or any(len(row) != width for row in levels):
        raise ValueError(f"glyph levels must be {width}x{height}")
    return pack_2bpp_values([value for row in levels for value in row])


def _candidate_from_metadata(stage26: dict, candidate: str | None) -> tuple[str, dict]:
    candidate_name = candidate or stage26["best_candidate"]
    candidates = stage26.get("candidates", {})
    if candidate_name not in candidates:
        raise ValueError(f"candidate {candidate_name!r} not found in Stage26 metadata")
    return candidate_name, candidates[candidate_name]


def export_stage26_nftr(
    *,
    source_nftr: Path = Path("a.NFTR"),
    stage26_metadata: Path = SONG13_LAYER_MLP_METADATA,
    candidate: str | None = None,
    out_nftr: Path | None = None,
    metadata_json: Path | None = None,
    preview_png: Path | None = None,
    preview_columns: int = 32,
    preview_scale: int = 4,
) -> Stage26NFTRExport:
    font = parse_rtfn_font(source_nftr)
    if font.bpp != 2:
        raise ValueError(f"{source_nftr} is {font.bpp}bpp; Stage26 NFTR export expects 2bpp")

    stage26 = json.loads(stage26_metadata.read_text(encoding="utf-8"))
    candidate_name, candidate_data = _candidate_from_metadata(stage26, candidate)
    glyphs = candidate_data["glyphs"]
    if len(glyphs) != font.glyph_count:
        raise ValueError(f"Stage26 glyph count {len(glyphs)} does not match NFTR glyph count {font.glyph_count}")

    out_dir = (out_nftr.parent if out_nftr else SONG13_LAYER_MLP_NFTR_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    nftr_path = out_nftr or out_dir / "a-stage26.NFTR"
    metadata_path = metadata_json or out_dir / "a-stage26.json"
    preview_path = preview_png or out_dir / "a-stage26-preview.png"

    patched = bytearray(font.data)
    for glyph in glyphs:
        index = int(glyph["index"])
        predicted_png = Path(glyph["predicted_png"])
        with Image.open(predicted_png) as image:
            if image.size != (font.cell_width, font.cell_height):
                raise ValueError(f"{predicted_png} is {image.size}, expected {(font.cell_width, font.cell_height)}")
            payload = pack_levels_2bpp(image_to_target_levels(image), font.cell_width, font.cell_height)
        if len(payload) != font.cell_size:
            raise ValueError(f"{predicted_png} packs to {len(payload)} bytes, expected {font.cell_size}")

        start = font.glyph_offset + index * font.cell_size
        patched[start : start + font.cell_size] = payload

    nftr_path.write_bytes(bytes(patched))
    export_font_atlas(nftr_path, preview_path, columns=preview_columns, scale=preview_scale)

    result = Stage26NFTRExport(
        source_nftr=str(source_nftr),
        stage26_metadata=str(stage26_metadata),
        candidate=candidate_name,
        out_nftr=str(nftr_path),
        metadata_json=str(metadata_path),
        preview_png=str(preview_path),
        glyph_count=font.glyph_count,
        cell_width=font.cell_width,
        cell_height=font.cell_height,
        bpp=font.bpp,
        bytes_per_glyph=font.cell_size,
        patched_glyph_count=len(glyphs),
    )
    payload = asdict(result) | {
        "source_contract": "Preserve original NFTR sections, widths, cmap, and glyph count; replace only PLGC 2bpp glyph payloads.",
        "stage26_candidate_metrics": {
            "cjk_quality_score": candidate_data.get("cjk_quality_score"),
            "groups": candidate_data.get("groups"),
            "foreground_ratio": candidate_data.get("foreground_ratio"),
            "source_contract": candidate_data.get("source_contract"),
        },
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
