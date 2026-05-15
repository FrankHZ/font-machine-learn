# Targets

This file records the current target split without the old stage-by-stage
research log.

Main objective:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

Layer semantics:

- `0`: transparent/background
- `1`: right-down shadow
- `2`: edge/anti-alias transition
- `3`: main stroke core

## Current Direction

The active source is WenQuanYi Bitmap Song 13px, rendered as:

```powershell
python scripts/render_source_glyphs.py --font fonts/WenQuanYi.Bitmap.Song.13px.ttf --font-index 0 --font-size 15 --font-mode L --threshold 96 --x-offset -1 --y-offset 1 --out-dir data/processed/glyphs/stage21_external_eval_sources/song13/source --metadata data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json --contact-sheet data/processed/glyphs/stage21_external_eval_sources/song13/source_target_contact.png
```

The source is Song/Ming style and has serif feet. Treat those as source shape,
not rendering errors.

Current modeling rule:

```text
preserve Song13 source shape -> assign 2/3 inside source -> add 1 outside source
```

Do not train Song13 stages to replace source glyph shape with target glyph
shape.

## Active Targets

### Target A: Keep Target Extraction Stable

Command:

```powershell
python scripts/extract_target_glyphs.py a.NFTR
```

Important facts:

- NFTR tags are reversed Nitro tags: `RTFN`, `FNIF`, `PLGC`, `HDWC`, `PAMC`
- cell: `15x15`
- bpp: `2`
- glyph count: `1814`

### Target B: Maintain Controlled Style Upper Bounds

The useful controlled sources are target-derived masks:

- `ge2`: target level `>=2`
- `eq3`: target level `==3`

Current Stage26 multi-source result:

| source | CJK visual |
|---|---:|
| target `>=2` | `0.9742` |
| target `==3` | `0.8903` |

Interpretation: `ge2` retains the edge information needed by the layer model.
`eq3` is too thin for this style.

### Target C: Preserve Song13 Source Shape

Stage25 is the current quality baseline:

- command: `python scripts/run_song13_source_locked.py`
- best rule: `edge_n1_diag_plus_right_from_source`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1220 / 0.8780`
- CJK visual: `0.6409`

Stage26 is the learned source-locked harness:

- command: `python scripts/train_song13_layer_mlp.py --jobs 4`
- best Song13 candidate: `core_patch_mlp_shadow_logistic_balanced_c055_s045`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1179 / 0.8821`
- CJK visual: `0.6339`

Interpretation: Stage26 is useful infrastructure but not a visual improvement
yet. Stage25 remains the Song13 quality reference.

### Target D: Speed Up Iteration Without Changing Metrics

Full multi-source Stage26 eval:

```powershell
python scripts/train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3
```

Quick subset eval:

```powershell
python scripts/train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3 --eval-limit 256 --search-limit 128
```

Use subset eval only for iteration. Omit `--eval-limit` for recorded metrics.

Parallel slow smokes:

```powershell
python scripts/run_slow_smokes.py --jobs 4
```

Stage27 review artifacts:

```powershell
python scripts/build_song13_review.py
```

This compares Stage24, Stage25, and Stage26 without making target visual score
the primary ranking signal.

## Historical Stage Summary

| stage | summary |
|---|---|
| 1 | target NFTR extraction |
| 2-10 | early WQY-shaped rules, MLPs, and visual metrics |
| 11 | formal 1bpp-visible to 2bpp dataset |
| 12 | target mask choice diagnostic (`ge2`, `eq3`) |
| 13-14 | WQY alignment and size diagnostics |
| 15-16 | controlled style MLP around `0.965` CJK visual |
| 17-19 | boundary and patch classifiers for `2/3` split |
| 20 | controlled two-head patch model, CJK visual `0.9739` |
| 21 | Song13 external transfer, CJK visual `0.6320` |
| 22-23 | target-shaped adapters improved metrics but deleted strokes |
| 24 | add-only adapter preserved strokes, visual `0.6508` |
| 25 | source-locked rule baseline, visual `0.6409` |
| 26 | source-locked learned layer harness, visual `0.6339` |
| 27 | human-review package for Stage24/25/26 |

## Next Useful Targets

- Improve Song13 shadow placement without deleting source pixels.
- Treat level `2` as edge/anti-alias, not generic stroke thickening.
- Compare new ideas against Stage25 by contact sheet first.
- Keep target-derived `ge2` as a controlled upper-bound sanity check.
- Keep full recorded metrics separate from subset iteration runs.
