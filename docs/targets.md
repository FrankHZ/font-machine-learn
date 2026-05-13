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

- target glyph PNGs under `data/processed/glyphs/stage1_target/target/`
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

- source glyph PNGs under `data/processed/glyphs/stage2_source/source/`
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

## Target 3: First Learning Baselines

Goal: test whether a small model can transform WQY Sharp bitmap glyphs into the
NFTR shadow style.

Keep the first baseline modest:

- no heavy dependency stack until Target 1 and Target 2 are stable
- preserve 2bpp target levels as class labels where practical
- compare model output against original glyphs by both pixels and contact sheets

Current baseline:

```powershell
python scripts/run_shadow_baseline.py
```

Rule:

- source ink: level `3`
- right/down neighbors: level `2`
- down-right shadow: level `1`

Metrics:

- per-glyph pixel accuracy
- per-glyph mean absolute level error
- per-glyph foreground IoU

Initial run:

- mean pixel accuracy: `0.5396`
- mean absolute level error: `0.8592`
- mean foreground IoU: `0.5461`

## Target 4: Trainable Pixel MLP

Goal: make the first trainable baseline without pulling in a heavy deep-learning
framework.

Command:

```powershell
python scripts/train_mlp_baseline.py
```

Model:

- framework: `scikit-learn`
- estimator: `MLPClassifier`
- features: `3x3` source patch, normalized `x/y`, and edge distance hints
- labels: target 2bpp level `0..3`

Outputs:

- `baseline_mlp/*.png`
- `baseline_mlp_metadata.json`
- `baseline_mlp_contact.png`

Initial run:

- train glyphs: `512`
- heldout glyphs: `1302`
- mean pixel accuracy: `0.6702`
- mean absolute level error: `0.6086`
- mean foreground IoU: `0.5515`
- heldout mean pixel accuracy: `0.6506`
- heldout mean absolute level error: `0.6368`
- heldout mean foreground IoU: `0.6081`

PyTorch or ONNX training should be added as a separate dependency decision once
this small trainable baseline is understood.

## Target 5: Visual Quality Scoring

Goal: make the harness care about the same things human review catches in the
contact sheets.

Command:

```powershell
python scripts/tune_shadow_baseline.py
```

Visual score inputs:

- level-3 ink F1
- level-1/2 shadow F1
- foreground IoU
- target-weighted level similarity
- isolated foreground noise penalty

Outputs:

- `baseline_tuned_shadow/*.png`
- `baseline_tuned_shadow_metadata.json`
- `baseline_tuned_shadow_search.json`
- `baseline_tuned_shadow_contact.png`

Initial run:

- best rule: `left_down_strong_diag_light`
- search-limit visual score: `0.4202`
- full mean visual score: `0.4919`
- full mean ink F1: `0.3506`
- full mean shadow F1: `0.3509`
- full mean foreground IoU: `0.6276`
- full mean pixel accuracy: `0.5339`
- full mean absolute level error: `0.7577`

This stage keeps rules explainable. The point is to rank shadow behavior by
visual resemblance before trying a heavier model again.

## Target 6: Human Review Artifacts

Goal: make visual review repeatable before changing model architecture.

Command:

```powershell
python scripts/build_review_report.py
```

Outputs:

- `review_report.json`
- `review_worst_cases.json`
- `review_contact.png`

The contact sheet uses a fixed horizontal order: source, rule shadow, tuned
shadow, MLP, target. Rows are selected from the worst visual-score cases so
failures are easy to inspect first.

Initial report:

- shadow visual score: `0.4659`
- tuned visual score: `0.4919`
- MLP visual score: `0.5288`
- note: the score still ranks MLP higher than human review does, so the review
  contact sheet remains the authority while the metric is refined.

## Target 7: 1bpp Target Diagnostic

Goal: answer whether current failures are mostly outline/alignment problems or
2bpp shade-level problems.

Command:

```powershell
python scripts/run_binary_diagnostic.py
```

Outputs:

- `target_1bpp/*.png`
- `binary_diagnostic_metadata.json`
- `binary_diagnostic_contact.png`

Metrics:

- binary pixel accuracy
- foreground precision
- foreground recall
- foreground F1
- foreground IoU
- false positive rate
- false negative rate

Use foreground F1 and foreground IoU as the main diagnostic values; binary pixel
accuracy is background-heavy and only supporting evidence.

Initial run:

- source F1/IoU: `0.4915` / `0.3146`
- shadow F1/IoU: `0.7028` / `0.5461`
- tuned F1/IoU: `0.7653` / `0.6276`
- MLP F1/IoU: `0.7065` / `0.5515`
- MLP precision/recall: `0.8164` / `0.6036`
- tuned precision/recall: `0.7396` / `0.7646`

Interpretation: MLP is conservative and misses foreground; tuned shadow better
matches the 1bpp target mask. The target mask is visibly heavier than the WQY
source, so Stage 8 should test source dilation/weight matching before another
model pass.

## Target 8: Source Weight Search

Goal: determine whether a simple global source dilation/offset improves the
foreground mask before 2bpp shadow synthesis.

Command:

```powershell
python scripts/search_weight_baseline.py
```

Search space:

- dilation kernels: `none`, `right_down`, `cardinal`, `box`
- offsets: `dx/dy` in `-1..1`
- score: 1bpp foreground F1, then foreground IoU

Outputs:

- `source_weighted/*.png`
- `baseline_weighted_shadow/*.png`
- `weight_search.json`
- `weight_search_metadata.json`
- `baseline_weighted_shadow_contact.png`

Initial run:

- best search rule: `box_dx-1_dy+1`
- weighted source F1/IoU: `0.7850` / `0.6490`
- weighted shadow visual score: `0.4559`
- weighted shadow foreground IoU: `0.6255`
- weighted shadow pixel accuracy: `0.5105`
- weighted shadow MAE: `1.1038`

Interpretation: global box dilation helps the 1bpp foreground mask but makes the
2bpp shadow output too heavy. The next pass should try a gentler or adaptive
weight rule rather than applying full dilation everywhere.

## Target 9: CJK-Focused Style Baseline

Goal: optimize the style baseline for CJK glyphs, with non-CJK glyphs acting as
regression guards instead of driving the search.

Command:

```powershell
python scripts/run_cjk_style_baseline.py
```

Style constraints:

- source core: level `3`
- source-adjacent edge/anti-alias transition: level `2`
- right-down `(1, 1)` shadow from source core: level `1`

Search and reporting:

- classify each glyph as `cjk`, `kana`, `latin`, `digit`, `punct`, `symbol`, or
  `unmapped`
- rank rules by CJK visual score, then CJK foreground F1/IoU
- report grouped metrics for `all`, `cjk`, and `non_cjk`
- generate a CJK-only contact sheet and a worst-CJK contact sheet

Outputs:

- `stage9_cjk_style/baseline_cjk_style/*.png`
- `stage9_cjk_style/cjk_style_metadata.json`
- `stage9_cjk_style/cjk_style_search.json`
- `stage9_cjk_style/cjk_style_contact.png`
- `stage9_cjk_style/cjk_style_worst_contact.png`

Initial run:

- CJK glyphs: `1528`
- best rule: `right_down_dx-1_dy+1`
- CJK visual score: `0.6357`
- CJK foreground F1/IoU: `0.8272` / `0.7112`
- CJK precision/recall: `0.8140` / `0.8424`
- all visual score: `0.5936`
- non-CJK visual score: `0.3682`

Interpretation: CJK-focused ranking selects a right/down edge transition rule
with the expected right-down level-1 shadow. It avoids Stage 8's over-heavy box
dilation while improving CJK foreground recall.

## Target 10: CJK Edge Transition Refinement

Goal: improve the placement of level `2` as an edge/anti-alias transition layer
rather than using it as generic stroke thickening.

Planned constraints:

- level `3`: source-derived main stroke core
- level `2`: constrained edge transition next to level `3`
- level `1`: fixed right-down `(1, 1)` shadow
- ranking remains CJK-first

Planned outputs:

- `stage10_cjk_edges/baseline_cjk_edges/*.png`
- `stage10_cjk_edges/cjk_edges_metadata.json`
- `stage10_cjk_edges/cjk_edges_search.json`
- `stage10_cjk_edges/cjk_edges_contact.png`
