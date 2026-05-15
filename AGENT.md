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

Stage26 original-layout NFTR export:

```powershell
python scripts/build_stage26_nftr.py
```

This export keeps the original `a.NFTR` sections, widths, cmap, and `1814`
glyph count. It only patches PLGC bitmap payloads from the Stage26 best
candidate. Do not confuse this with the full-map `ds_nftr/a.txt` build.

Stage26 full-map NFTR export:

```powershell
python scripts/build_stage26_full_nftr.py
```

This reads `ds_nftr/a.txt` (`CODE=char`), retrains the Stage26 heads, and
rebuilds a `3296`-glyph NFTR. Preserve the old `ds_nftr` rules: reuse
Latin/digits/punct and `一二三` from the original NFTR when present, substitute
`… -> ‥`, left-bottom align generated CJK, and give `，；` padded advance (`6`).

Stage28 target quantization calibration:

```powershell
python scripts/run_target_quantized_calibration.py
```

This tests target-derived `visible`, `>=2`, and `==3` 1bpp source masks. Current
CJK learned visual scores are `0.6125`, `0.9684`, and `0.8873`; `>=2` is the
right target-shaped source mask before trying convolutional models.

Stage29 lightweight convolution-feature probe:

```powershell
python scripts/run_target_conv_calibration.py
```

Current best is `core_conv_mlp_shadow_conv_mlp_c055_s045`, CJK visual `0.9604`
with ink/shadow F1 `0.9623 / 0.9315`. This is below Stage28 `>=2` learned
`0.9684`, so hand-built small convolution features are not enough by themselves.

Stage30 tiny PyTorch CNN target probe:

```powershell
python scripts/run_target_torch_cnn.py
```

Current config is 48 channels, 200 epochs, LR `0.003`, source-locked inference.
On CUDA (`torch 2.12.0+cu130`, RTX 3080 Ti), CJK visual is `0.9790`,
ink/shadow F1 `0.9824 / 0.9640`, beating Stage28 and Stage29 on target-shaped
source masks.

CUDA install used in this workspace:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

Use `--eval-limit` only for quick iteration, not for recorded metrics.

Stage31 Song13 PyTorch CNN transfer:

```powershell
python scripts/run_song13_torch_cnn.py
```

This trains the same target `>=2` tiny CNN and applies it to the Song13 1bpp
source mask with source-locked inference. Current CUDA CJK visual is `0.6332`,
source deletion `0.0000`, and source level `2/3` is about `0.1805 / 0.8195`.
Treat it as a CNN transfer harness and contact-sheet candidate, not a metric win
over Stage25/26.

Stage32 public comparison contact:

```powershell
python scripts/build_public_comparison_contact.py
```

This writes `docs/assets/stage32_public_comparison.png` for README display. It
uses rows `source`, `stage25`, `stage26`, and `stage32`; the `stage32` row is the
Stage31 torch-transfer output. Do not include the target NFTR row in public
README assets.

Release package:

```powershell
python scripts/build_release_bmfont.py
```

This builds a full selected-font-cmap AngelCode BMFont package under `release/`.
Do not use `ds_nftr/a.txt` for the default release package; that map is
game-specific. The release format is `.fnt + RGBA PNG atlas + JSON`, not BDF/PCF,
because the output needs shadow and edge levels.

## Important Files

- `a.NFTR`: local-only decompressed target NFTR, ignored for public release.
  Reversed Nitro tags: `RTFN`, `FNIF`, `PLGC`, `HDWC`, `PAMC`.
- `src/font_machine_learn/nftr.py`: parser/exporter.
- `src/font_machine_learn/paths.py`: canonical generated-output paths.
- `src/font_machine_learn/song13_source_locked.py`: Stage25.
- `src/font_machine_learn/song13_layer_mlp.py`: Stage26.
- `src/font_machine_learn/stage26_nftr.py`: Stage26 predicted PNGs -> original-layout NFTR.
- `src/font_machine_learn/stage26_full_nftr.py`: Stage26 full-map NFTR builder.
- `src/font_machine_learn/song13_review.py`: Stage27 human-review artifacts.
- `src/font_machine_learn/target_quantized_calibration.py`: Stage28 target mask calibration.
- `src/font_machine_learn/target_conv_calibration.py`: Stage29 lightweight conv probe.
- `src/font_machine_learn/target_torch_cnn.py`: Stage30 tiny PyTorch CNN probe.
- `src/font_machine_learn/song13_torch_cnn.py`: Stage31 target-trained CNN -> Song13 transfer.
- `scripts/build_public_comparison_contact.py`: Stage32 public README comparison.
- `docs/assets/stage32_public_comparison.png`: promoted public comparison image.
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
| `stage26_song13_layer_mlp/nftr` | Stage26 original-layout NFTR export |
| `stage27_song13_review` | Stage24/25/26 review contact sheets |
| `stage28_target_quantized_calibration` | target 1bpp quantization calibration |
| `stage29_target_ge2_conv` | lightweight convolution-feature target probe |
| `stage30_target_ge2_torch` | tiny PyTorch CNN target probe |
| `stage31_song13_torch_cnn` | target-trained CNN transferred to Song13 |
| `docs/assets` | promoted public README image assets |

## Working Rules

- Keep scripts thin and reusable logic under `src/font_machine_learn/`.
- Keep default tests fast; put full stage exports behind slow smokes.
- Prefer contact sheets for visual decisions.
- Track CJK as the primary split and non-CJK as a guard split.
- Treat target-shaped metrics as diagnostics for Song13, not the final truth.
- Never delete source pixels in Song13 source-locked stages.
- Use Stage27 sheets before judging a new Song13 model by target visual score.
- Do not commit `a.NFTR`, generated NFTRs, or target glyph extraction artifacts.
- Public README images should omit the target NFTR row.
