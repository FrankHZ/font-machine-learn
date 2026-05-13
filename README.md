# Font Machine Learn

This project is an experiment for learning the visual style of a Nintendo DS
bitmap font: small 15 x 15 px glyphs with 2bpp levels for transparent,
shadow, edge, and main stroke pixels.

The first milestone is intentionally small:

1. read the available `a.NFTR` font asset;
2. export a PNG atlas for visual inspection;
3. keep enough metadata around to make later dataset extraction repeatable.

## Current Layout

```text
.
├── a.NFTR                         # Source font file currently available
├── AGENT.md                       # Working notes for coding agents
├── README.md
├── pyproject.toml                 # Python project metadata and tool config
├── requirements.txt               # Minimal runtime dependency list
├── docs/
│   └── targets.md                 # Milestones and dataset boundaries
│   └── deliverables.md            # Stage deliverables and verification
├── requirements-ml.txt            # CPU-friendly ML/data dependencies
├── scripts/
│   ├── check_env.py               # Python/package environment check
│   └── export_nftr.py             # CLI wrapper for exporting an atlas
├── src/
│   └── font_machine_learn/
│       ├── __init__.py
│       └── nftr.py                # RTFN/NFTR parser and PNG exporter
├── tests/
│   └── test_nftr_export.py        # Smoke-test harness
└── data/
    ├── raw/                       # Put original extracted assets here later
    ├── interim/                   # Parsed glyph metadata, debug artifacts
    └── processed/                 # Atlases and training-ready outputs
```

`a.NFTR` is kept at the repository root for now because it is the only real
asset. Once more assets exist, move originals into `data/raw/`.

## Python Setup

Python 3.13 is known to run in this workspace. Create a virtual environment if
you want isolation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pillow is the only runtime dependency for the first milestone.

Install the CPU-friendly data/ML stack when preparing datasets or baselines:

```powershell
python -m pip install -r requirements-ml.txt
python scripts/check_env.py
```

## Harness

Run the smoke-test harness with:

```powershell
python -m unittest discover
```

The test exports `a.NFTR` into a temporary directory and verifies the key source
font facts: `RTFN` mode, `15x15`, `2bpp`, `57` bytes per glyph, and `1814`
glyphs.

## Export the Font Atlas

Run:

```powershell
python scripts/export_nftr.py a.NFTR --out data/processed/a_atlas.png
```

The command writes:

- `data/processed/a_atlas.png`
- `data/processed/a_atlas.json`

The JSON records the detected format and glyph layout. The current `a.NFTR`
is a game-compatible Nitro NFTR variant with reversed section tags:

- header: `RTFN`
- info: `FNIF`
- glyph bitmap: `PLGC`
- width table: `HDWC`
- character map: `PAMC`

Known fields from the source font:

- cell size: `15x15`
- bpp: `2`
- bytes per glyph: `57`
- baseline: `15`
- glyph count: `1814`

## Notes on Pixel Levels

The pixels are not ordinary anti-aliased grayscale. Treat the four levels as
semantic bitmap layers:

- `0`: transparent/background
- `1`: shadow
- `2`: occasional edge/intermediate color
- `3`: main stroke

The exporter also has a raw fallback for diagnostic work with incorrectly
extracted or compressed files:

```powershell
python scripts/export_nftr.py some.raw --cell-width 15 --cell-height 15 --bpp 2 --offset 0
```

## Next Milestones

See `docs/targets.md` for the working target split.
See `docs/deliverables.md` for stage deliverables and commit checkpoints.

1. Extract individual glyph PNGs plus labels/codepoints from `PAMC` mappings.
2. Pair source monochrome bitmap fonts with target shaded glyphs.
3. Train a small image-to-image model or rule-assisted model for 15 x 15 glyphs.
4. Compare generated glyphs against the original NFTR levels and in-game previews.
