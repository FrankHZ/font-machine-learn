from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFont
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.baseline import levels_to_image  # noqa: E402
from font_machine_learn.paths import TARGET_METADATA  # noqa: E402
from font_machine_learn.song13_torch_cnn import train_target_ge2_cnn  # noqa: E402
from font_machine_learn.stage26_full_nftr import load_mapping_entries, render_mask, width_from_levels  # noqa: E402
from font_machine_learn.target_torch_cnn import predict_levels  # noqa: E402


DEFAULT_OUT_DIR = Path("release/font-machine-learn-stage32-song13-bmfont")
CONTROL_CODES = set(range(0x00, 0x20)) | set(range(0x7F, 0xA0))


def mask_to_tensor(mask: list[list[bool]]) -> torch.Tensor:
    values = np.asarray(mask, dtype=np.float32)[None, :, :]
    return torch.from_numpy(values)


def load_font_cmap_chars(font_path: Path, font_index: int) -> list[str]:
    font = TTFont(str(font_path), fontNumber=font_index)
    codes: set[int] = set()
    for table in font["cmap"].tables:
        if table.isUnicode():
            codes.update(int(code) for code in table.cmap)
    chars: list[str] = []
    for code in sorted(codes):
        if code in CONTROL_CODES or 0xD800 <= code <= 0xDFFF:
            continue
        try:
            chars.append(chr(code))
        except ValueError:
            continue
    return chars


def load_charset(
    *,
    font_path: Path,
    font_index: int,
    charset_map: Path | None,
) -> tuple[str, list[str]]:
    if charset_map is not None:
        entries = load_mapping_entries(charset_map, 0xE800)
        return f"map:{charset_map}", [char for _code, char in entries]
    return f"font-cmap:{font_path}#{font_index}", load_font_cmap_chars(font_path, font_index)


def write_bmfont(
    *,
    path: Path,
    atlas_file: str,
    glyphs: list[dict],
    scale_w: int,
    scale_h: int,
    line_height: int,
    base: int,
) -> None:
    lines = [
        'info face="Font Machine Learn Stage32" size=15 bold=0 italic=0 charset="unicode" unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=1,1',
        f"common lineHeight={line_height} base={base} scaleW={scale_w} scaleH={scale_h} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4",
        f'page id=0 file="{atlas_file}"',
        f"chars count={len(glyphs)}",
    ]
    for glyph in glyphs:
        lines.append(
            "char "
            f"id={glyph['unicode']} "
            f"x={glyph['x']} y={glyph['y']} "
            f"width={glyph['width']} height={glyph['height']} "
            f"xoffset=0 yoffset=0 xadvance={glyph['xadvance']} "
            "page=0 chnl=15"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release_bmfont(
    *,
    target_metadata: Path,
    charset_map: Path | None,
    font_path: Path,
    out_dir: Path,
    package_name: str,
    font_index: int,
    font_size: int,
    font_mode: str,
    threshold: int,
    x_offset: int,
    y_offset: int,
    cell_width: int,
    cell_height: int,
    columns: int,
    channels: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    make_zip: bool,
) -> dict:
    if not target_metadata.exists():
        raise FileNotFoundError(f"Target metadata not found: {target_metadata}. Run target extraction locally first.")
    target = json.loads(target_metadata.read_text(encoding="utf-8"))
    charset_source, chars = load_charset(font_path=font_path, font_index=font_index, charset_map=charset_map)
    font = ImageFont.truetype(str(font_path), size=font_size, index=font_index)
    model, losses, device, cuda_device = train_target_ge2_cnn(
        list(target["glyphs"]),
        channels=channels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    rows = math.ceil(len(chars) / columns)
    atlas_w = columns * cell_width
    atlas_h = rows * cell_height
    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    glyphs: list[dict] = []
    glyph_dir = out_dir / "glyphs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, char in enumerate(chars):
        mask = render_mask(
            font,
            char,
            cell_width=cell_width,
            cell_height=cell_height,
            font_mode=font_mode,
            threshold=threshold,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        levels = predict_levels(model, mask_to_tensor(mask))
        image = levels_to_image(levels)
        col = index % columns
        row = index // columns
        x = col * cell_width
        y = row * cell_height
        atlas.alpha_composite(image, (x, y))
        width = width_from_levels(char, levels, cell_width)
        glyph_png = glyph_dir / f"u{ord(char):04X}.png"
        image.save(glyph_png)
        glyphs.append(
            {
                "index": index,
                "char": char,
                "unicode": ord(char),
                "x": x,
                "y": y,
                "width": cell_width,
                "height": cell_height,
                "xadvance": width.advance,
                "glyph_png": str(glyph_png.relative_to(out_dir)),
            }
        )

    atlas_name = f"{package_name}.png"
    fnt_name = f"{package_name}.fnt"
    json_name = f"{package_name}.json"
    atlas_path = out_dir / atlas_name
    fnt_path = out_dir / fnt_name
    json_path = out_dir / json_name
    atlas.save(atlas_path)
    write_bmfont(
        path=fnt_path,
        atlas_file=atlas_name,
        glyphs=glyphs,
        scale_w=atlas_w,
        scale_h=atlas_h,
        line_height=cell_height,
        base=cell_height,
    )
    metadata = {
        "name": package_name,
        "format": "AngelCode BMFont text + RGBA PNG atlas",
        "license": "GPL-2.0-only",
        "target_metadata": str(target_metadata),
        "charset_source": charset_source,
        "charset_map": str(charset_map) if charset_map is not None else "",
        "font": {
            "path": str(font_path),
            "index": font_index,
            "size": font_size,
            "mode": font_mode,
            "threshold": threshold,
            "x_offset": x_offset,
            "y_offset": y_offset,
        },
        "model": {
            "source": "Stage31/Stage32 target-trained tiny torch CNN",
            "channels": channels,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "device": str(device),
            "cuda_device": cuda_device,
            "final_loss": losses[-1] if losses else 0.0,
        },
        "glyph_count": len(glyphs),
        "cell_width": cell_width,
        "cell_height": cell_height,
        "columns": columns,
        "atlas": atlas_name,
        "bmfont": fnt_name,
        "glyphs": glyphs,
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    release_readme = out_dir / "README.md"
    release_readme.write_text(
        "\n".join(
            [
                "# Font Machine Learn Stage32 BMFont 字体包",
                "",
                "格式：AngelCode BMFont 文本 `.fnt` + RGBA PNG 图集。",
                "",
                "文件：",
                f"- `{fnt_name}`: BMFont metrics 和 Unicode 映射",
                f"- `{atlas_name}`: 2bpp 风格 RGBA 字形图集",
                f"- `{json_name}`: 构建参数和逐字记录",
                "- `glyphs/`: 单字 PNG",
                "",
                "使用：读取 `.fnt` 中的 `char id/x/y/width/height/xadvance`，从 PNG 图集中裁出 glyph，按 `xadvance` 排版。",
                "",
                "渲染建议：使用 nearest-neighbor，不要重新抗锯齿或线性缩放。",
                "",
                "授权：GPL-2.0-only。本包由文泉驿字体生成。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if Path("COPYRIGHT.md").exists():
        shutil.copy2("COPYRIGHT.md", out_dir / "COPYRIGHT.md")
    if Path("LICENSES/GPL-2.0.txt").exists():
        license_dir = out_dir / "LICENSES"
        license_dir.mkdir(exist_ok=True)
        shutil.copy2("LICENSES/GPL-2.0.txt", license_dir / "GPL-2.0.txt")

    zip_path = out_dir.with_suffix(".zip")
    if make_zip:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except PermissionError:
                zip_path = out_dir.with_name(f"{out_dir.name}-new").with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.rglob("*")):
                archive.write(path, path.relative_to(out_dir.parent))
    return {
        "out_dir": str(out_dir),
        "zip": str(zip_path) if make_zip else "",
        "atlas": str(atlas_path),
        "bmfont": str(fnt_path),
        "metadata": str(json_path),
        "glyph_count": len(glyphs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full-glyph Stage32 BMFont release package.")
    parser.add_argument("--target-metadata", type=Path, default=TARGET_METADATA)
    parser.add_argument("--charset-map", type=Path, default=None, help="Optional CODE=char map. Defaults to the selected font cmap.")
    parser.add_argument("--font", type=Path, default=Path("fonts/WenQuanYi.Bitmap.Song.13px.ttf"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--package-name", default="font-machine-learn-stage32-song13")
    parser.add_argument("--font-index", type=int, default=0)
    parser.add_argument("--font-size", type=int, default=15)
    parser.add_argument("--font-mode", default="L")
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--x-offset", type=int, default=-1)
    parser.add_argument("--y-offset", type=int, default=1)
    parser.add_argument("--cell-width", type=int, default=15)
    parser.add_argument("--cell-height", type=int, default=15)
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--random-seed", type=int, default=32)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    result = build_release_bmfont(
        target_metadata=args.target_metadata,
        charset_map=args.charset_map,
        font_path=args.font,
        out_dir=args.out_dir,
        package_name=args.package_name,
        font_index=args.font_index,
        font_size=args.font_size,
        font_mode=args.font_mode,
        threshold=args.threshold,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        columns=args.columns,
        channels=args.channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        make_zip=not args.no_zip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
