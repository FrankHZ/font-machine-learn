# Font Machine Learn

This project learns the style of a Nintendo DS `15x15` 2bpp bitmap font.

The task is:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

Layer semantics:

- `0`: transparent/background
- `1`: right-down shadow
- `2`: edge or anti-alias transition
- `3`: main stroke core

The current production-facing source is WenQuanYi Bitmap Song 13px rendered into
the same `15x15` cell. The project no longer treats target glyphs as shapes to
copy. Current Song13 work preserves source pixels first, then assigns style
layers around that fixed source shape.

## Quick Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python scripts/check_env.py
```

For CUDA PyTorch on this Windows/NVIDIA setup:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

Fast verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Slow stage smokes:

```powershell
.\.venv\Scripts\python.exe scripts\run_slow_smokes.py --jobs 4
.\.venv\Scripts\python.exe scripts\run_slow_smokes.py --jobs 3 --pattern song13
```

The parallel runner launches each `@slow_test` as a separate `python -B -m
unittest` process with isolated temp directories. The `song13` subset currently
runs in about `79s` with `--jobs 3`, compared with about `189s` summed
individual test time.

## Current Commands

Export the target NFTR:

```powershell
.\.venv\Scripts\python.exe scripts\extract_target_glyphs.py a.NFTR
```

Render the current Song13 source baseline:

```powershell
.\.venv\Scripts\python.exe scripts\render_source_glyphs.py --font fonts/WenQuanYi.Bitmap.Song.13px.ttf --font-index 0 --font-size 15 --font-mode L --threshold 96 --x-offset -1 --y-offset 1 --out-dir data/processed/glyphs/stage21_external_eval_sources/song13/source --metadata data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json --contact-sheet data/processed/glyphs/stage21_external_eval_sources/song13/source_target_contact.png
```

Run the clean source-locked rule baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_song13_source_locked.py
```

Run the learned source-locked layer model:

```powershell
.\.venv\Scripts\python.exe scripts\train_song13_layer_mlp.py --jobs 4
```

Run one training pass and evaluate Song13 plus target-derived controls:

```powershell
.\.venv\Scripts\python.exe scripts\train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3
```

Quick iteration version:

```powershell
.\.venv\Scripts\python.exe scripts\train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3 --eval-limit 256 --search-limit 128
```

Use `--eval-limit` only for quick checks. Omit it for recorded metrics.

Build Song13 human-review artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\build_song13_review.py
```

Calibrate target 1bpp quantization before larger models:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_quantized_calibration.py
```

Probe lightweight convolution features on target `>=2` source masks:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_conv_calibration.py
```

Train the tiny PyTorch CNN target probe:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_torch_cnn.py
```

Transfer the target-trained tiny CNN to Song13 source masks:

```powershell
.\.venv\Scripts\python.exe scripts\run_song13_torch_cnn.py
```

Build a game-facing NFTR from the current Stage26 best candidate:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage26_nftr.py
```

This preserves the original `a.NFTR` sections, widths, cmap, and `1814` glyph
count. It replaces only PLGC glyph bitmap payloads with Stage26 predicted 2bpp
pixels. Use the full-map command below for the `ds_nftr/a.txt` build.

Build the current full-map candidate from `ds_nftr/a.txt`:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage26_full_nftr.py
```

This retrains the Stage26 layer heads, renders all `3296` chars from the
`CODE=char` map, reuses original NFTR glyphs/widths for Latin, digits,
punctuation, `一二三`, and `… -> ‥`, and writes a rebuilt NFTR with the original
map codes. `，` and `；` receive padded advance (`6`) so they do not crowd the
following glyph without taking a full cell.

## Current Findings

Song13 render contract:

- font: `fonts/WenQuanYi.Bitmap.Song.13px.ttf`
- size: `15`
- mode: `L`
- threshold: `96`
- offset: `x=-1`, `y=+1`
- the face is Song/Ming style; serif feet are expected

Stage25 is the current Song13 quality baseline:

- rule: `edge_n1_diag_plus_right_from_source`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1220 / 0.8780`
- CJK visual: `0.6409`
- ink/shadow F1: `0.5721 / 0.5483`

Stage26 is the learned harness, not a quality win yet:

- best Song13 candidate: `core_patch_mlp_shadow_logistic_balanced_c055_s045`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1179 / 0.8821`
- CJK visual: `0.6339`
- ink/shadow F1: `0.5759 / 0.5315`

Stage26 multi-source controls after one shared training pass:

| source | CJK visual |
|---|---:|
| Song13 | `0.6339` |
| target `>=2` | `0.9742` |
| target `==3` | `0.8903` |

Interpretation: the layer model works very well when the source shape is the
target `>=2` mask. Song13 remains lower because its glyph shape differs from the
NFTR target. Target `==3` is too thin because it discards the level-2 edge
information.

Stage28 makes that target-quantized control explicit. CJK learned visual scores:

| target source mask | flat3 | learned |
|---|---:|---:|
| `visible` | `0.5835` | `0.6125` |
| `>=2` | `0.6091` | `0.9684` |
| `==3` | `0.6183` | `0.8873` |

Interpretation: `>=2` is the useful 1bpp target source. `visible` bakes shadow
into the source, and `==3` throws away too much edge information.

Stage29 tests a small convolution-feature probe on the target `>=2` source mask:

- best candidate: `core_conv_mlp_shadow_conv_mlp_c055_s045`
- CJK visual: `0.9604`
- ink/shadow F1: `0.9623 / 0.9315`

Interpretation: this is below Stage28 `>=2` learned visual `0.9684`. Small
hand-built convolution features are not enough to beat the wider patch MLP.

Stage30 trains a tiny PyTorch CNN on the same target `>=2` source mask:

- model: 3 Conv3x3 ReLU blocks, 48 channels, source-locked inference
- CJK visual: `0.9790`
- ink/shadow F1: `0.9824 / 0.9640`

Interpretation: real convolution does beat the patch MLP on target-shaped source.

Stage31 applies the Stage30-style CNN to Song13 source masks:

- model: same 48-channel tiny CNN, trained on target `>=2`
- CJK visual: `0.6344`
- source deletion: `0.0000`
- source level `2/3`: `0.1578 / 0.8422`

Interpretation: this is a useful CNN transfer harness, but it does not beat the
human-reviewed Stage25/26 Song13 candidates by metric. Contact-sheet review is
more important than target overlap here because Song13 and the target NFTR have
different glyph shapes.

## Repository Layout

```text
.
├── a.NFTR
├── fonts/
├── scripts/
├── src/font_machine_learn/
├── tests/
├── docs/
└── data/processed/glyphs/
```

Generated glyph artifacts live under stage folders in
`data/processed/glyphs/`. Keep generated PNG/JSON out of the stage root; each
stage owns its own directory.

Important stage folders:

| stage | purpose |
|---|---|
| `stage1_target` | split NFTR target glyphs |
| `stage12_target_masks` | target `>=2` and `==3` masks |
| `stage20_shadow_classifier` | current controlled target-derived best |
| `stage21_external_eval_sources/song13` | current Song13 source render |
| `stage24_song13_add_only` | source-preserving adapter diagnostic |
| `stage25_song13_source_locked` | current rule quality baseline |
| `stage26_song13_layer_mlp` | learned source-locked layer harness |
| `stage26_song13_layer_mlp/nftr` | Stage26 original-layout NFTR export |
| `stage27_song13_review` | human-review sheets for Stage24/25/26 |
| `stage28_target_quantized_calibration` | target 1bpp quantization calibration |
| `stage29_target_ge2_conv` | lightweight convolution-feature target probe |
| `stage30_target_ge2_torch` | tiny PyTorch CNN target probe |
| `stage31_song13_torch_cnn` | target-trained CNN transferred to Song13 |

## Stage Summary

| stage | result |
|---|---|
| 1 | NFTR target glyph dataset exported |
| 2-10 | early WQY-shaped baselines and visual metrics |
| 11 | formal 1bpp-visible to 2bpp dataset |
| 12-13 | target mask and WQY alignment diagnostics |
| 15-16 | controlled style MLP; `ge2` around `0.965` visual |
| 17-19 | explainable and patch-based `2/3` core-edge split |
| 20 | controlled two-head patch model, CJK visual `0.9739` |
| 21 | Song13 external transfer, CJK visual `0.6320` |
| 22-23 | target-shaped adapters improved metrics but deleted strokes |
| 24 | add-only adapter preserves strokes, visual `0.6508` |
| 25 | source-locked rule baseline, visual `0.6409` |
| 26 | source-locked learned layer harness, visual `0.6339`; original-layout and full-map NFTR exports available |
| 27 | human-review package comparing Stage24/25/26 |
| 28 | target quantized calibration; `>=2` learned CJK visual `0.9684` |
| 29 | target `>=2` lightweight conv probe; CJK visual `0.9604` |
| 30 | target `>=2` tiny torch CNN; CUDA CJK visual `0.9790` |
| 31 | Song13 tiny torch CNN transfer; CJK visual `0.6344`, source deletion `0.0000` |

## Notes

- Do not use synthetic boldening as a default for tiny bitmap fonts.
- Do not judge Song13 only by target pixel overlap; the shapes differ.
- Contact sheets remain the primary review artifact.
- Use CJK as the primary split and non-CJK as a guard split.
- For full-map NFTR builds, use `ds_nftr/a.txt` and preserve old `ds_nftr`
  rules: Latin/punct reuse, `… -> ‥`, `一二三` simple-stroke reuse, CJK
  left-bottom alignment, and padded advance for `，；`.
- Keep generated artifacts disposable unless a stage explicitly promotes them.
