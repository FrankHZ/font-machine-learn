# Agent Notes

This repo experiments with tiny bitmap font style transfer:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

Layer meanings:

- `0`: transparent/background
- `1`: right-down shadow
- `2`: edge/anti-alias transition
- `3`: main stroke core

Do not frame the main work as target glyph-shape correction. For Song13 work,
source pixels are the contract.

## Current Source Contract

Default external source is WenQuanYi Bitmap Song 13px:

```powershell
python scripts/render_source_glyphs.py --font fonts/WenQuanYi.Bitmap.Song.13px.ttf --font-index 0 --font-size 15 --font-mode L --threshold 96 --x-offset -1 --y-offset 1 --out-dir data/processed/glyphs/stage21_external_eval_sources/song13/source --metadata data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json --contact-sheet data/processed/glyphs/stage21_external_eval_sources/song13/source_target_contact.png
```

Notes:

- This is Song/Ming style; serif feet are expected.
- Threshold is not a meaningful tuning axis for this bitmap strike.
- Avoid broad font comparisons unless explicitly requested.
- Do not default to dilation/boldening for tiny bitmap fonts.

## Current Baselines

Stage25 is the current quality reference:

- script: `scripts/run_song13_source_locked.py`
- output: `data/processed/glyphs/stage25_song13_source_locked/`
- best rule: `edge_n1_diag_plus_right_from_source`
- source deleted ratio: `0.0000`
- CJK visual: `0.6409`

Stage26 is the learned source-locked harness:

- script: `scripts/train_song13_layer_mlp.py`
- output: `data/processed/glyphs/stage26_song13_layer_mlp/`
- best Song13 candidate: `core_patch_mlp_shadow_logistic_balanced_c055_s045`
- source deleted ratio: `0.0000`
- CJK visual: `0.6339`

Stage26 controls:

- Song13 visual: `0.6339`
- target `>=2` visual: `0.9742`
- target `==3` visual: `0.8903`

Interpretation: the learned layerer works when source shape matches target
`>=2`. Song13 remains lower because source shape differs. Stage25 remains the
quality baseline until the learned model visibly beats it.

## Fast Commands

Default tests:

```powershell
python -m unittest discover
```

Parallel slow smoke tests:

```powershell
python scripts/run_slow_smokes.py --jobs 4
python scripts/run_slow_smokes.py --jobs 3 --pattern song13
```

The runner executes each `@slow_test` in its own `python -B -m unittest`
process with `FML_RUN_SLOW_TESTS=1` and `PYTHONDONTWRITEBYTECODE=1`. The
`song13` subset has been checked at about `79s` with `--jobs 3`, versus about
`189s` summed individual test time.

Stage26 full multi-source eval:

```powershell
python scripts/train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3
```

Stage26 quick multi-source eval:

```powershell
python scripts/train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3 --eval-limit 256 --search-limit 128
```

Use `--eval-limit` only for quick iteration, not for recorded metrics.

## Important Files

- `a.NFTR`: decompressed target NFTR. Reversed Nitro tags: `RTFN`, `FNIF`,
  `PLGC`, `HDWC`, `PAMC`.
- `src/font_machine_learn/nftr.py`: parser/exporter.
- `src/font_machine_learn/paths.py`: canonical generated-output paths.
- `src/font_machine_learn/song13_source_locked.py`: Stage25.
- `src/font_machine_learn/song13_layer_mlp.py`: Stage26.
- `tests/test_nftr_export.py`: fast tests plus slow stage smokes.
- `docs/targets.md`: current target split and findings.
- `docs/deliverables.md`: verification and stage checkpoint summary.

## Output Layout Rules

Generated artifacts must live under `data/processed/glyphs/stageN_name/`.

Do not add generated PNG/JSON directly under `data/processed/glyphs/`. For a new
stage:

- add constants to `src/font_machine_learn/paths.py`;
- make script defaults use those constants;
- document the stage in README, `docs/targets.md`, and
  `docs/deliverables.md`;
- keep contact sheet order explicit in metadata.

Important folders:

| folder | purpose |
|---|---|
| `stage1_target` | target glyph PNGs and metadata |
| `stage12_target_masks` | target `>=2` and `==3` masks |
| `stage20_shadow_classifier` | controlled target-derived best baseline |
| `stage21_external_eval_sources/song13` | current Song13 source |
| `stage25_song13_source_locked` | rule quality baseline |
| `stage26_song13_layer_mlp` | learned source-locked layer harness |

## Working Rules

- Keep scripts thin and reusable logic under `src/font_machine_learn/`.
- Keep default tests fast; put full stage exports behind slow smokes.
- Prefer contact sheets for visual decisions.
- Track CJK as the primary split and non-CJK as a guard split.
- Treat target-shaped metrics as diagnostics for Song13, not the final truth.
- Never delete source pixels in Song13 source-locked stages.
