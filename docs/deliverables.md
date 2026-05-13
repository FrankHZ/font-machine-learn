# Deliverables

Each stage should end with a runnable artifact, a quick verification command,
and a commit. Generated PNG/JSON outputs are disposable unless a stage explicitly
promotes them to documented fixtures.

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
- note fallback-box glyphs separately before training

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
- document PyTorch/ONNX as a later, separately pinned dependency decision
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
