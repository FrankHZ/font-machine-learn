# Targets

This file keeps the project split into concrete targets so future work does not
drift into format guessing or premature ML work.

## Target 0: Source Font Export

Status: working.

- Input: `a.NFTR`
- Format: reversed-tag Nitro NFTR (`RTFN`, `FNIF`, `PLGC`, `HDWC`, `PAMC`)
- Glyph cells: `15x15`
- Pixel format: `2bpp`
- Output: atlas PNG plus JSON metadata under `data/processed/`
- Harness: `python -m unittest discover`

## Target 1: Glyph Dataset Extraction

Goal: split the source NFTR into labeled per-glyph records.

Expected outputs:

- target glyph PNGs under `data/processed/glyphs/target/`
- machine-readable metadata with glyph index, code, decoded character when
  possible, width metrics, and pixel-level histogram
- a small contact sheet for quick visual review

Command:

```powershell
python scripts/extract_target_glyphs.py a.NFTR
```

Key detail: labels should come from `PAMC` mappings, not only atlas position.

## Target 2: Source Bitmap Rendering

Goal: render the full source character set from `wqy-zenhei.ttc`, using the
WQY Sharp face as the source bitmap font to be transformed.

Known preferred settings from previous work:

- font file: `wqy-zenhei.ttc`
- face index: `2`
- size: `13`
- target cell: `15x15`

Expected outputs:

- source glyph PNGs under `data/processed/glyphs/source/`
- paired source/target metadata for training
- preview contact sheets showing source, target, and simple baseline transforms

Command:

```powershell
python scripts/render_source_glyphs.py
```

Expected shape:

- TTC index `2` resolves to `WenQuanYi Zen Hei Sharp`
- current source render has `1293 / 1812` non-empty glyphs at `12-13px` ink width
- symbols not covered by WQY Sharp may render as fallback boxes and should be
  flagged or reused from the original NFTR in later pairing work

## Target 3: First Learning Baseline

Goal: test whether a small model can transform WQY Sharp bitmap glyphs into the
NFTR shadow style.

Keep the first baseline modest:

- no heavy dependency stack until Target 1 and Target 2 are stable
- preserve 2bpp target levels as class labels where practical
- compare model output against original glyphs by both pixels and contact sheets
