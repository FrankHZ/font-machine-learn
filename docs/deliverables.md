# Deliverables

Each stage should end with:

- a runnable command;
- generated metadata and contact sheets under `data/processed/glyphs/stageN_*`;
- a fast verification path;
- a commit.

Default verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Parallel slow-smoke verification:

```powershell
.\.venv\Scripts\python.exe scripts\run_slow_smokes.py --jobs 4
.\.venv\Scripts\python.exe scripts\run_slow_smokes.py --jobs 3 --pattern song13
```

The runner splits `@slow_test` methods across worker processes. The `song13`
subset currently completes in about `79s` with `--jobs 3`, compared with about
`189s` summed individual test time.

## Current Deliverables

### Stage25: Source-Locked Song13 Rules

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_song13_source_locked.py
```

Outputs:

- `data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_metadata.json`
- `data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_search.json`
- `data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_contact.png`
- `data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_error_contact.png`

Current CJK metrics:

- best rule: `edge_n1_diag_plus_right_from_source`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1220 / 0.8780`
- visual: `0.6409`
- ink/shadow F1: `0.5721 / 0.5483`

Status: current Song13 visual quality baseline.

### Stage26: Source-Locked Layer MLP

Full command:

```powershell
.\.venv\Scripts\python.exe scripts\train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3
```

Quick iteration command:

```powershell
.\.venv\Scripts\python.exe scripts\train_song13_layer_mlp.py --eval-target-quantized --eval-source-jobs 3 --eval-limit 256 --search-limit 128
```

Outputs:

- `data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_metadata.json`
- `data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_contact.png`
- `data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_error_contact.png`
- optional multi-source evals under `stage26_song13_layer_mlp/eval_sources/`

NFTR export command:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage26_nftr.py
```

NFTR export outputs:

- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26.NFTR`
- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26.json`
- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26-preview.png`

This export preserves the original NFTR layout and patches only PLGC glyph
payload bytes from the Stage26 predicted PNGs.

Full-map NFTR export command:

```powershell
.\.venv\Scripts\python.exe scripts\build_stage26_full_nftr.py
```

Full-map NFTR export outputs:

- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26-fullmap.NFTR`
- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26-fullmap.json`
- `data/processed/glyphs/stage26_song13_layer_mlp/nftr/a-stage26-fullmap-preview.png`

The full-map build consumes `ds_nftr/a.txt` (`CODE=char`) and currently produces
`3296` glyphs. `，` and `；` use padded advance (`6`) even though their ink is
smaller; this avoids punctuation crowding without taking a full cell.

Current Song13 CJK metrics:

- best candidate: `core_patch_mlp_shadow_logistic_balanced_c055_s045`
- source deleted ratio: `0.0000`
- source level `2/3`: `0.1179 / 0.8821`
- visual: `0.6339`
- ink/shadow F1: `0.5759 / 0.5315`

Current controlled CJK metrics:

| source | visual |
|---|---:|
| Song13 | `0.6339` |
| target `>=2` | `0.9742` |
| target `==3` | `0.8903` |

Status: useful learned harness, not a quality win over Stage25 yet.

### Stage27: Song13 Human Review Package

Command:

```powershell
.\.venv\Scripts\python.exe scripts\build_song13_review.py
```

Outputs:

- `data/processed/glyphs/stage27_song13_review/song13_review_metadata.json`
- `data/processed/glyphs/stage27_song13_review/song13_review_overview.png`
- category sheets for representative, simple, complex, disagreement, heavy
  shadow, high gray2, and hole-fill-risk glyphs

Image order:

```text
source -> stage24 adapted -> stage24 predicted -> stage25 predicted -> stage26 predicted -> target
```

Status: review/eval artifact, not a training stage.

### Stage28: Target Quantized Calibration

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_quantized_calibration.py
```

Outputs:

- `data/processed/glyphs/stage28_target_quantized_calibration/target_quantized_calibration_metadata.json`
- per-mode folders for `visible`, `ge2`, and `eq3`
- per-mode contact sheets with image order:

```text
source mask -> flat3 -> learned 2bpp -> target 2bpp
```

Current CJK visual scores:

| target source mask | flat3 | learned |
|---|---:|---:|
| `visible` | `0.5835` | `0.6125` |
| `>=2` | `0.6091` | `0.9684` |
| `==3` | `0.6183` | `0.8873` |

Status: calibration stage. `>=2` remains the source-shape upper bound to use
before trying convolutional models.

### Stage29: Target GE2 Lightweight Conv Probe

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_conv_calibration.py
```

Outputs:

- `data/processed/glyphs/stage29_target_ge2_conv/target_ge2_conv_metadata.json`
- `data/processed/glyphs/stage29_target_ge2_conv/target_ge2_conv_contact.png`
- `data/processed/glyphs/stage29_target_ge2_conv/target_ge2_conv_errors.png`

Image order:

```text
source ge2 -> predicted 2bpp -> target 2bpp
```

Current CJK metrics:

- best candidate: `core_conv_mlp_shadow_conv_mlp_c055_s045`
- visual: `0.9604`
- ink/shadow F1: `0.9623 / 0.9315`

Status: probe stage. It does not beat Stage28 `>=2` learned visual `0.9684`;
small hand-built convolution features lose some information versus wider patch
features.

### Stage30: Target GE2 Tiny PyTorch CNN

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_target_torch_cnn.py
```

Outputs:

- `data/processed/glyphs/stage30_target_ge2_torch/target_ge2_torch_metadata.json`
- `data/processed/glyphs/stage30_target_ge2_torch/target_ge2_torch_contact.png`
- `data/processed/glyphs/stage30_target_ge2_torch/target_ge2_torch_errors.png`

Image order:

```text
source ge2 -> predicted 2bpp -> target 2bpp
```

Current CJK metrics:

- model: 3 Conv3x3 ReLU blocks, 48 channels, source-locked inference
- epochs/LR: `200 / 0.003`
- device: CUDA, `torch 2.12.0+cu130`, RTX 3080 Ti
- visual: `0.9790`
- ink/shadow F1: `0.9824 / 0.9640`

Status: target-shaped source probe. It beats Stage28 `>=2` learned visual
`0.9684` and Stage29 `0.9604`, so true convolution is worth carrying forward.

### Stage31: Song13 Tiny PyTorch CNN Transfer

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_song13_torch_cnn.py
```

Outputs:

- `data/processed/glyphs/stage31_song13_torch_cnn/song13_torch_cnn_metadata.json`
- `data/processed/glyphs/stage31_song13_torch_cnn/song13_torch_cnn_contact.png`
- `data/processed/glyphs/stage31_song13_torch_cnn/song13_torch_cnn_errors.png`

Image order:

```text
original source -> source ge2 -> predicted 2bpp -> target 2bpp
```

Current CJK metrics:

- model: Stage30-style 3 Conv3x3 ReLU blocks, 48 channels
- training source: target level `>=2`
- eval source: Song13 1bpp render
- epochs/LR: `200 / 0.003`
- device: CUDA, RTX 3080 Ti
- visual: `0.6332`
- ink/shadow F1: `0.5632 / 0.5420`
- source deletion: `0.0000`
- source level `2/3`: `0.1805 / 0.8195`

Status: useful CNN transfer harness. It keeps the Song13 source intact and gives
slightly more level-2 edge than Stage25/26, but it is not a metric win. Judge by
contact sheet before deciding whether to build a Stage31 NFTR.

### Stage32: Public Comparison Contact

Command:

```powershell
.\.venv\Scripts\python.exe scripts\build_public_comparison_contact.py
```

Outputs:

- `docs/assets/stage32_public_comparison_song13.png`
- `docs/assets/stage32_public_comparison_song12.png`
- `docs/assets/stage32_public_comparison.json`

Image order:

```text
source -> stage25 -> stage26 -> stage32
```

Status: public README artifact. The `stage32` row uses Stage31 torch-transfer
output, and the sheet intentionally omits the target NFTR row.

### Release Package: Full Font BMFont

Command:

```powershell
.\.venv\Scripts\python.exe scripts\build_release_bmfont.py
```

Outputs:

- `release/font-machine-learn-stage32-song13-fullcmap.zip`
- `release/font-machine-learn-stage32-song12-fullcmap.zip`

Status: release artifact. It enumerates the selected WenQuanYi font cmap rather
than `ds_nftr/a.txt`, so users do not need to regenerate a game-specific map.

## Historical Checkpoints

| stage | deliverable |
|---|---|
| 0 | venv, git, fast unittest harness |
| 1 | target glyph extraction |
| 2 | WQY source rendering |
| 3-5 | rule/MLP/shadow visual baselines |
| 6 | review contact sheets |
| 7-8 | binary target and source-weight diagnostics |
| 9-10 | CJK-focused style rules |
| 11 | 1bpp-to-2bpp paired dataset |
| 12-14 | target mask and WQY size/alignment diagnostics |
| 15-16 | controlled style MLP baselines |
| 17-20 | boundary, patch, and shadow classifiers |
| 21 | external Song13 transfer eval |
| 22-23 | target-shaped Song13 adapters |
| 24 | add-only source-preserving adapter |
| 25 | source-locked rule baseline |
| 26 | source-locked learned layer harness |
| 27 | Song13 human-review package |
| 28 | target quantized calibration |
| 29 | target `>=2` lightweight conv probe |
| 30 | target `>=2` tiny PyTorch CNN |
| 31 | Song13 tiny PyTorch CNN transfer |
| 32 | public Stage25/26/32 comparison contact |

## Commit Themes

Use concise commit themes:

- `feat:` for new stages or model behavior
- `perf:` for runtime/harness speedups
- `test:` for harness changes
- `docs:` for documentation-only updates

Generated artifacts are disposable unless explicitly promoted. Commit code and
docs; do not commit bulk stage PNGs unless requested.
