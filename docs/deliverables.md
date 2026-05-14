# Deliverables

Each stage should end with a runnable artifact, a quick verification command,
and a commit. Generated PNG/JSON outputs are disposable unless a stage explicitly
promotes them to documented fixtures.

Core objective:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

WQY Sharp stages are useful diagnostics for the future real input font, but the
main model should learn style layering from 1bpp masks to 2bpp labels, not glyph
shape correction.

## Stage 0: Project Harness

Deliverable:

- Python virtual environment created locally as `.venv/`
- dependencies installed from `requirements.txt` and `requirements-ml.txt`
- Git repository initialized
- source NFTR atlas export covered by `python -m unittest discover`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Default `unittest discover` is a fast harness. Full stage-export smoke tests are
behind `FML_RUN_SLOW_TESTS=1` so normal verification does not rebuild every
historical artifact.

Commit theme:

```text
chore: initialize font learning harness
```

## Stage 1: Target Glyph Dataset

Deliverable:

- split `a.NFTR` into per-glyph target PNGs
- write metadata with glyph index, Shift-JIS code, decoded char when possible,
  width metrics, and 2bpp level histogram
- produce a contact sheet for review
- command: `scripts/extract_target_glyphs.py`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: export target glyph dataset
```

## Stage 2: WQY Source Glyph Dataset

Deliverable:

- render matching source glyphs from `wqy-zenhei.ttc`
- use WQY Sharp face index `2`, size `13`, cell `15x15`
- write paired source/target metadata
- produce source/target comparison contact sheets
- command: `scripts/render_source_glyphs.py`
- note fallback-box glyphs separately before using WQY as a future real input

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: render wqy source glyph dataset
```

## Stage 3: Baseline Transform

Deliverable:

- first non-deep-learning or tiny-model baseline
- output predicted 2bpp glyphs
- compare predictions to target glyphs with pixel metrics and contact sheets
- command: `scripts/run_shadow_baseline.py`
- baseline rule: source ink is level `3`; right/down pixels are level `2`;
  down-right shadow is level `1`
- first recorded metrics: pixel accuracy `0.5396`, MAE `0.8592`,
  foreground IoU `0.5461`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add rule-based shadow baseline
```

## Stage 4: Trainable Baseline Stack

Deliverable:

- train a small pixel-level model without adding a heavy framework yet
- use `scikit-learn` MLP over local source-pixel patches and pixel coordinates
- output predicted 2bpp glyphs, metrics, and contact sheet
- command: `scripts/train_mlp_baseline.py`
- document PyTorch/ONNX as a later, separately pinned dependency decision after
  the 1bpp-to-2bpp data contract is stable
- first recorded metrics: pixel accuracy `0.6702`, MAE `0.6086`,
  foreground IoU `0.5515`; heldout pixel accuracy `0.6506`, heldout MAE
  `0.6368`, heldout foreground IoU `0.6081`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add trainable mlp baseline
```

## Stage 5: Visual Quality Scoring

Deliverable:

- add metrics that do not let background pixels dominate the score
- measure level-3 ink F1, level-1/2 shadow F1, foreground IoU, weighted
  similarity, and isolated foreground noise
- search a small set of explainable shadow rules with the visual score
- output tuned glyphs, a search report, metrics, and a contact sheet
- command: `scripts/tune_shadow_baseline.py`
- first recorded best rule: `left_down_strong_diag_light`
- first recorded full-run metrics: visual score `0.4919`, ink F1 `0.3506`,
  shadow F1 `0.3509`, foreground IoU `0.6276`, pixel accuracy `0.5339`,
  MAE `0.7577`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add visual shadow tuning
```

## Stage 6: Review Artifacts

Deliverable:

- build a side-by-side contact sheet for human review
- compare source, rule shadow, tuned shadow, MLP, and target in a fixed order
- write a summary JSON for baseline-level visual metrics
- write a worst-case JSON sorted by weakest visual score
- command: `scripts/build_review_report.py`
- first report visual scores: shadow `0.4659`, tuned `0.4919`, MLP `0.5288`
- known caveat: the visual score still overrates MLP compared with human review

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add baseline review artifacts
```

## Stage 7: 1bpp Target Diagnostic

Deliverable:

- quantize NFTR target glyphs to foreground/background
- score source, rule shadow, tuned shadow, and MLP as binary masks
- report binary accuracy, foreground precision/recall/F1, foreground IoU, false
  positive rate, and false negative rate
- output 1bpp target glyphs, metadata, and a worst-case contact sheet
- establish target-derived 1bpp masks as the source side of the main style
  learning dataset
- command: `scripts/run_binary_diagnostic.py`
- first recorded F1/IoU: source `0.4915`/`0.3146`, shadow `0.7028`/`0.5461`,
  tuned `0.7653`/`0.6276`, MLP `0.7065`/`0.5515`
- first interpretation: MLP is high-precision but low-recall; tuned shadow
  better matches target foreground, and source glyphs likely need weight/dilation

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add binary target diagnostics
```

## Stage 8: Source Weight Search

Deliverable:

- search a small global grid of source glyph offsets and dilation kernels
- choose the best source transformation by 1bpp foreground F1/IoU
- export weighted source glyphs
- apply the tuned shadow rule to the weighted source and export a contact sheet
- command: `scripts/search_weight_baseline.py`
- treat this as a WQY input diagnostic, not the primary training objective
- first recorded best rule: `box_dx-1_dy+1`
- first recorded weighted source F1/IoU: `0.7850` / `0.6490`
- first recorded weighted shadow visual score: `0.4559`, foreground IoU
  `0.6255`, pixel accuracy `0.5105`, MAE `1.1038`
- first interpretation: global box dilation improves binary mask recall but is
  visually too heavy for 2bpp shadow output

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add source weight search
```

## Stage 9: CJK-Focused Style Baseline

Deliverable:

- classify glyphs into `cjk`, `kana`, `latin`, `digit`, `punct`, `symbol`, or
  `unmapped`
- search style candidates using CJK metrics first
- keep non-CJK metrics as regression guard summaries
- enforce source core as level `3`, source-adjacent edge transition pixels as
  level `2`, and fixed right-down `(1, 1)` shadow as level `1`
- output CJK-only and worst-CJK contact sheets
- command: `scripts/run_cjk_style_baseline.py`
- first recorded best rule: `right_down_dx-1_dy+1`
- first recorded CJK metrics: visual score `0.6357`, F1 `0.8272`, IoU
  `0.7112`, precision `0.8140`, recall `0.8424`
- first recorded guard metrics: all visual `0.5936`, non-CJK visual `0.3682`

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: add cjk-focused style baseline
```

## Stage 10: CJK Edge Transition Refinement

Deliverable:

- treat level `2` as edge/anti-alias transition, not stroke thickening
- refine where level `2` may appear around level `3` core pixels
- search transition direction and density on CJK glyphs first
- keep fixed right-down `(1, 1)` level `1` shadow
- compare CJK contact sheets before changing any model code
- command: `scripts/run_cjk_edges_baseline.py`
- first recorded best rule: `right_n8_dx-1_dy+1`
- first recorded CJK metrics: visual score `0.6387`, F1 `0.8295`, IoU
  `0.7149`, precision `0.8441`, recall `0.8170`
- first interpretation: right-only level-2 transition is cleaner than
  right/down transition for CJK while fixed right-down level-1 shadow remains

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: refine cjk edge transitions
```

## Stage 11: 1bpp-to-2bpp Style Dataset

Deliverable:

- build the formal paired dataset for the real task
- input: target-derived 1bpp visible glyph masks
- label: original NFTR 2bpp glyph levels
- include `char_class` and CJK/non-CJK grouping
- run a rule baseline that keeps the silhouette fixed and splits visible pixels
  into core `3`, edge transition `2`, and right-down shadow `1`
- output source masks, label references, baseline predictions, metadata, and a
  contact sheet
- command: `scripts/build_1bpp_style_dataset.py`
- first recorded best rule: `core_n3_right_down_dx-1_dy-1`
- first recorded CJK metrics: visual score `0.8170`, foreground F1 `1.0000`,
  foreground IoU `1.0000`
- first recorded guard metrics: all visual `0.8257`, non-CJK visual `0.8718`
- first interpretation: foreground metrics are perfect by construction because
  the baseline no longer edits glyph shape; review layer assignment by visual
  score and contact sheets

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
feat: build 1bpp style dataset
```

## Stage 12: Target Mask Choice Diagnostic

Deliverable:

- export two target-derived 1bpp masks: target level `>=2` and target level
  `==3`
- compare both masks against WQY Sharp source glyphs
- report grouped metrics for all, CJK, non-CJK, source-nonempty, and
  CJK-source-nonempty glyphs
- output a contact sheet with WQY source, `>=2`, `==3`, and original 2bpp target
- command: `scripts/compare_target_masks_to_source.py`
- first recorded CJK `>=2` F1/IoU: `0.4141` / `0.2657`
- first recorded CJK `==3` F1/IoU: `0.3903` / `0.2517`
- first interpretation: `>=2` is closer to WQY Sharp than `==3`, but neither is
  a clean final source definition. Level `2` behaves like anti-alias/edge
  information, so quantizing it upward or downward destroys part of the target
  effect, and WQY/NFTR shape alignment remains weak.

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Full stage smoke:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_nftr_export.NFTRExportTest.test_compares_target_masks_to_wqy_source_smoke
```

Commit theme:

```text
feat: compare target mask choices
```

## Stage 13: WQY Alignment and Weight Diagnostic

Deliverable:

- search WQY source offset and weight rules against three target mask views:
  `visible`, `>=2`, and `==3`
- rank rules by CJK source-nonempty foreground F1/IoU
- export aligned WQY masks and matching target masks for each target view
- output a contact sheet showing source, aligned/target pairs, and original 2bpp
  target
- command: `scripts/run_wqy_alignment_diagnostic.py`
- first recorded CJK source-nonempty `visible` best rule:
  `cardinal_dx+0_dy+1`, F1/IoU `0.8282` / `0.7111`
- first recorded CJK source-nonempty `>=2` best rule:
  `right_down_dx-1_dy+0`, F1/IoU `0.6388` / `0.4674`
- first recorded CJK source-nonempty `==3` best rule:
  `right_down_dx-1_dy+0`, F1/IoU `0.5861` / `0.4232`
- first interpretation: WQY is closer to the full visible silhouette after
  adding weight, but it is not a clean core or core+edge source; source
  normalization and style-layer learning should remain separate.

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Full stage smoke:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_nftr_export.NFTRExportTest.test_exports_wqy_alignment_diagnostic_smoke
```

Commit theme:

```text
feat: diagnose wqy alignment
```

## Stage 14: WQY Sharp Size-14 Diagnostic

Deliverable:

- render WQY Sharp face index `2` at size `14` in the existing `15x15` cell
- compare raw size-14 source masks against target `>=2` and `==3` masks
- do not apply synthetic dilation/boldening
- output source glyphs, source/target contact sheet, target mask comparison
  metadata, and target mask contact sheet under `stage14_wqy_size14/`
- first recorded CJK source-nonempty size-14 `>=2` F1/IoU: `0.5590` / `0.4078`
- first recorded CJK source-nonempty size-14 `==3` F1/IoU: `0.5326` / `0.3815`
- first interpretation: size `14` is a better raw WQY source than size `13`
  without violating the tiny-bitmap-font constraint against artificial
  thickening

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
docs: record wqy size 14 diagnostic
```

## Stage 15: Controlled 1bpp-to-2bpp Style MLP

Deliverable:

- train a small pixel-level MLP on target-derived 1bpp inputs and original NFTR
  2bpp labels
- compare `visible`, `>=2`, and `==3` controlled source modes
- select the best controlled mode by CJK visual score
- run WQY13 and WQY14 as external inputs with the best controlled model
- output per-mode source masks, 2bpp predictions, metadata, and contact sheets
- command: `scripts/train_style_mlp.py`
- first recorded best controlled mode: `ge2`
- first recorded CJK controlled visual scores: `visible` `0.9031`, `ge2`
  `0.9651`, `eq3` `0.9633`
- first recorded CJK external visual scores: WQY13 `0.4707`, WQY14 `0.6029`
- first interpretation: controlled style learning works; WQY transfer remains a
  source-normalization problem

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Full stage smoke:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_nftr_export.NFTRExportTest.test_exports_style_mlp_smoke
```

Commit theme:

```text
feat: train controlled style mlp
```

## Stage 16: Style MLP Tuning and Ablation

Deliverable:

- rerun the controlled style MLP with slightly larger capacity
- record whether edge/distance features or two-head classifiers are worth
  promoting
- keep the result honest: compare against Stage 15 and call out marginal gains
- command: `scripts/train_style_mlp.py --hidden-units 128 --max-iter 100 --random-seed 17 --out-dir data/processed/glyphs/stage16_style_mlp_tuning --metadata data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_metadata.json --contact-sheet data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_contact.png`
- first recorded best controlled mode: `ge2`
- first recorded CJK controlled visual score: `0.9654`
- first recorded CJK external visual scores: WQY13 `0.4677`, WQY14 `0.5995`
- first interpretation: larger MLP capacity gives only a tiny controlled
  improvement; edge-feature and two-head ablations did not beat Stage 15

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Commit theme:

```text
docs: record style mlp tuning
```

## Stage 17: Explainable Level-2 Boundary Rules

Deliverable:

- search simple rules for `ge2` source pixels: level `2` boundary versus level
  `3` core
- report feature statistics for target level `2` and `3`
- export rule predictions, search JSON, metadata, CJK contact sheet, and
  worst-CJK contact sheet
- command: `scripts/run_boundary_rules.py`
- first recorded best rule: `n2_band0_plain`
- first recorded CJK metrics: visual score `0.9361`, ink F1 `0.9355`, shadow F1
  `0.8877`, foreground IoU `0.9515`
- first interpretation: target level `2` is overwhelmingly a boundary pixel,
  but simple geometry is not enough to match the MLP

Verification:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Full stage smoke:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_nftr_export.NFTRExportTest.test_exports_boundary_rules_smoke
```

Commit theme:

```text
feat: explain ge2 boundary rules
```
