# Targets

This file keeps the project split into concrete targets so future work does not
drift into format guessing or premature ML work.

Main objective:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

The project is not primarily about changing glyph shapes. WQY Sharp is a future
real input font and a useful diagnostic source. The cleanest paired learning
setup is target-derived 1bpp masks as input and original NFTR 2bpp glyphs as
labels.

Layer semantics:

- `0`: transparent/background
- `1`: fixed right-down shadow
- `2`: edge/anti-alias transition around the main stroke
- `3`: main stroke core from the input 1bpp mask

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
WQY Sharp face as the future real 1bpp input font to validate the style pipeline.

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

## Target 3: First WQY-Shaped Baselines

Goal: test whether simple rules can apply NFTR-like style layers to WQY-shaped
1bpp input. This is diagnostic for the future real input font, not the final
training formulation.

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

## Target 4: Trainable Pixel MLP Diagnostic

Goal: make the first trainable baseline without pulling in a heavy deep-learning
framework. This stage proved the training harness worked, but it used WQY-shaped
input and should not define the final model objective.

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

PyTorch or ONNX training should be added only after the main 1bpp-to-2bpp data
contract is stable.

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

Goal: quantize target glyphs into foreground/background. This began as a
diagnostic, but it also creates the correct source side for the main paired
learning setup.

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
matches the 1bpp target mask. More importantly, target-derived 1bpp masks should
become the canonical source representation for learning 2bpp style layering.

## Target 8: Source Weight Search

Goal: determine whether a simple global source dilation/offset improves the
foreground mask before 2bpp shadow synthesis. This remains a WQY diagnostic for
future production input, not the primary training target.

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
regression guards instead of driving the search. This still uses WQY-shaped
input and should be treated as style-rule exploration.

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
rather than using it as generic stroke thickening. This stage clarifies the
style semantics that the main 1bpp-to-2bpp model should learn.

Command:

```powershell
python scripts/run_cjk_edges_baseline.py
```

Constraints:

- level `3`: source-derived main stroke core
- level `2`: constrained edge transition next to level `3`
- level `1`: fixed right-down `(1, 1)` shadow
- ranking remains CJK-first

Search space:

- edge offset sets: `none`, `right`, `down`, `right_down`, `horizontal`,
  `vertical`, `cardinal`
- core-neighbor limits: `1`, `2`, `3`, `8`
- offsets: `dx/dy` in `-1..1`

Outputs:

- `stage10_cjk_edges/baseline_cjk_edges/*.png`
- `stage10_cjk_edges/cjk_edges_metadata.json`
- `stage10_cjk_edges/cjk_edges_search.json`
- `stage10_cjk_edges/cjk_edges_contact.png`
- `stage10_cjk_edges/cjk_edges_worst_contact.png`

Initial run:

- CJK glyphs: `1528`
- best rule: `right_n8_dx-1_dy+1`
- CJK visual score: `0.6387`
- CJK foreground F1/IoU: `0.8295` / `0.7149`
- CJK precision/recall: `0.8441` / `0.8170`
- all visual score: `0.5946`
- non-CJK visual score: `0.3590`

Interpretation: compared with Stage 9, the best rule removes the down level-2
edge and keeps only a right-side transition plus the fixed right-down shadow.
This slightly improves CJK visual score and precision while reducing recall.

## Target 11: 1bpp-to-2bpp Style Dataset

Goal: pivot the main pipeline to the real learning problem: given a complete
1bpp glyph mask, generate the NFTR-style 2bpp layer assignment.

Inputs and labels:

- input: target-derived `1bpp` visible mask, later replaceable by any complete
  1bpp font
- label: original NFTR target glyph with levels `0..3`
- primary subset: CJK
- guard subsets: kana, latin, digit, punctuation, symbol

Command:

```powershell
python scripts/build_1bpp_style_dataset.py
```

Planned outputs:

- `stage11_1bpp_style/input_1bpp/*.png`
- label references to Stage 1 target glyphs
- `stage11_1bpp_style/baseline_2bpp/*.png`
- `stage11_1bpp_style/style_pairs_metadata.json`
- `stage11_1bpp_style/style_baseline_search.json`
- `stage11_1bpp_style/style_baseline_contact.png`
- `stage11_1bpp_style/style_baseline_worst_contact.png`

Baseline rule:

- keep the 1bpp visible silhouette fixed
- infer a dense level `3` core inside that silhouette
- assign remaining visible pixels to level `2` edge/transition by default
- allow visible non-core pixels right-down from core to become level `1` shadow

Initial run:

- CJK glyphs: `1528`
- best rule: `core_n3_right_down_dx-1_dy-1`
- CJK visual score: `0.8170`
- CJK foreground F1/IoU: `1.0000` / `1.0000`
- all visual score: `0.8257`
- non-CJK visual score: `0.8718`

Interpretation: Stage 11 is now evaluating layer assignment rather than glyph
shape alignment. Perfect foreground F1/IoU is expected because the baseline does
not change the input silhouette; the useful signal is visual score and contact
sheets showing where core/edge/shadow splitting is wrong.

## Target 12: Target Mask Choice Diagnostic

Goal: decide which target-derived 1bpp definition is closest to WQY Sharp source
glyphs before using it as a model input convention.

Command:

```powershell
python scripts/compare_target_masks_to_source.py
```

Mask choices:

- `ge2`: target levels `2` and `3`
- `eq3`: target level `3` only

Outputs:

- `stage12_target_masks/target_ge2_1bpp/*.png`
- `stage12_target_masks/target_eq3_1bpp/*.png`
- `stage12_target_masks/target_mask_compare_metadata.json`
- `stage12_target_masks/target_mask_compare_contact.png`

Contact sheet order:

- WQY source
- target `>=2`
- target `==3`
- original target 2bpp

Initial run:

- CJK source-nonempty glyphs: `1528`
- CJK `ge2` F1/IoU: `0.4141` / `0.2657`
- CJK `eq3` F1/IoU: `0.3903` / `0.2517`
- CJK `ge2` precision/recall: `0.4340` / `0.3831`
- CJK `eq3` precision/recall: `0.3667` / `0.3977`

Interpretation: target `>=2` is slightly closer to WQY Sharp than target `==3`,
but this is a diagnostic result, not a final source contract. Level `2` appears
to carry edge/anti-alias information; forcing it into a binary source either
thickens the source or makes it too thin. The low scores are also expected
because WQY and NFTR differ in glyph shape, placement, and core alignment.

Stage 13 should avoid treating target quantization as a solved input. Better
next directions are:

- keep `>=2` and `==3` as diagnostic masks, not authoritative labels
- train/evaluate level assignment on true 2bpp labels instead of collapsing
  level `2`
- use contact sheets and level-aware visual metrics as the main signal
- separately estimate WQY-to-target alignment/weight before judging style
  transfer quality

## Target 13: WQY Alignment and Weight Diagnostic

Goal: measure how much of the WQY gap is explained by simple source offset and
stroke weight before asking a style model to learn 2bpp layers.

Command:

```powershell
python scripts/run_wqy_alignment_diagnostic.py
```

Search:

- source transform: Stage 8 weight rules (`none`, `right_down`, `cardinal`,
  `box`) with `dx/dy` in `-1..1`
- target views: `visible` (`level > 0`), `ge2` (`level >= 2`), and `eq3`
  (`level == 3`)
- ranking: CJK source-nonempty foreground F1, then IoU

Outputs:

- `stage13_wqy_alignment/aligned/{visible,ge2,eq3}/*.png`
- `stage13_wqy_alignment/aligned/target_{visible,ge2,eq3}/*.png`
- `stage13_wqy_alignment/wqy_alignment_metadata.json`
- `stage13_wqy_alignment/wqy_alignment_search.json`
- `stage13_wqy_alignment/wqy_alignment_contact.png`

Initial CJK source-nonempty run:

- `visible`: best `cardinal_dx+0_dy+1`, F1/IoU `0.8282` / `0.7111`,
  precision/recall `0.7557` / `0.9216`
- `ge2`: best `right_down_dx-1_dy+0`, F1/IoU `0.6388` / `0.4674`,
  precision/recall `0.5124` / `0.8653`
- `eq3`: best `right_down_dx-1_dy+0`, F1/IoU `0.5861` / `0.4232`,
  precision/recall `0.4380` / `0.8976`

Interpretation: WQY can approximate the full visible NFTR silhouette after
adding weight, but it does not align cleanly with target core or core+edge masks.
The best `visible` rule is a heavy cardinal dilation, so it is compensating for
source weight/shape rather than learning NFTR style. Keep WQY adaptation as a
separate source-normalization problem before training the 2bpp style model.

Important follow-up: for tiny bitmap fonts, synthetic dilation should not be a
default source-normalization path. Prefer checking native bitmap strike sizes and
placement first.

## Target 14: WQY Sharp Size-14 Diagnostic

Goal: test whether WQY Sharp size `14` is a better raw 1bpp source in the same
`15x15` cell than the previous size `13`, without synthetic boldening.

Commands:

```powershell
python scripts/render_source_glyphs.py --font-size 14 --out-dir data/processed/glyphs/stage14_wqy_size14/source --metadata data/processed/glyphs/stage14_wqy_size14/source_metadata.json --contact-sheet data/processed/glyphs/stage14_wqy_size14/source_target_contact.png
python scripts/compare_target_masks_to_source.py --source-metadata data/processed/glyphs/stage14_wqy_size14/source_metadata.json --ge2-dir data/processed/glyphs/stage14_wqy_size14/target_ge2_1bpp --eq3-dir data/processed/glyphs/stage14_wqy_size14/target_eq3_1bpp --metadata data/processed/glyphs/stage14_wqy_size14/target_mask_compare_metadata.json --contact-sheet data/processed/glyphs/stage14_wqy_size14/target_mask_compare_contact.png
```

Initial CJK source-nonempty comparison:

- size `13` raw `ge2` F1/IoU: `0.4141` / `0.2657`
- size `13` raw `eq3` F1/IoU: `0.3903` / `0.2517`
- size `14` raw `ge2` F1/IoU: `0.5590` / `0.4078`
- size `14` raw `eq3` F1/IoU: `0.5326` / `0.3815`

Size-14 width profile:

- non-empty glyphs: `1812`
- glyphs with ink width `12..13`: `1458`
- glyphs with ink width `14..15`: `4`
- average ink width: `12.06`
- max ink width: `14`

Interpretation: size `14` is a better raw WQY source candidate than size `13`
and improves alignment without artificial thickening. Keep dilation results as
diagnostic upper bounds only, not a production target.

## Target 15: Controlled 1bpp-to-2bpp Style MLP

Goal: train the first model for the corrected task while keeping WQY shape
differences outside the training objective.

Command:

```powershell
python scripts/train_style_mlp.py
```

Setup:

- model: `sklearn.neural_network.MLPClassifier`
- features: `5x5` source-mask patch, normalized coordinates, neighbor counts
- labels: original NFTR 2bpp levels `0..3`
- train glyphs: first `768`
- controlled source modes: `visible` (`>0`), `ge2` (`>=2`), `eq3` (`==3`)
- external sources: WQY13 and WQY14, evaluated with the best controlled model

Initial CJK controlled results:

- `visible`: visual `0.9031`, ink F1 `0.8543`, shadow F1 `0.8614`,
  foreground IoU `1.0000`
- `ge2`: visual `0.9651`, ink F1 `0.9639`, shadow F1 `0.9443`,
  foreground IoU `0.9717`
- `eq3`: visual `0.9633`, ink F1 `1.0000`, shadow F1 `0.9282`,
  foreground IoU `0.9284`

Initial CJK external results using the best controlled `ge2` model:

- WQY13 visual `0.4707`, ink F1 `0.3831`, shadow F1 `0.3481`,
  foreground IoU `0.5595`
- WQY14 visual `0.6029`, ink F1 `0.5430`, shadow F1 `0.4877`,
  foreground IoU `0.6602`

Interpretation: style-layer learning is feasible under controlled target-derived
inputs. `ge2` and `eq3` both work well, with `ge2` slightly ahead on CJK visual
score. WQY14 transfers better than WQY13, but external scores remain much lower,
so WQY source adaptation should stay separate from the 2bpp style model.

## Target 16: Style MLP Tuning and Ablation

Goal: test whether Stage 15 can improve by simple capacity/feature tuning before
moving to a different model family.

Command:

```powershell
python scripts/train_style_mlp.py --hidden-units 128 --max-iter 100 --random-seed 17 --out-dir data/processed/glyphs/stage16_style_mlp_tuning --metadata data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_metadata.json --contact-sheet data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_contact.png
```

Initial CJK controlled comparison:

- Stage 15 `ge2`: visual `0.9651`, ink F1 `0.9639`, shadow F1 `0.9443`
- Stage 16 tuned `ge2`: visual `0.9654`, ink F1 `0.9656`, shadow F1 `0.9439`
- Stage 16 tuned `eq3`: visual `0.9649`, ink F1 `1.0000`, shadow F1 `0.9328`

Initial CJK external comparison with the tuned `ge2` model:

- WQY13 visual `0.4677`
- WQY14 visual `0.5995`

Ablation note:

- hard source constraints were already learned by Stage 15; postprocessing made
  `0` pixel changes
- a two-head tree classifier performed poorly (`visual ~0.71`)
- edge/distance features plus heavier level-2 sample weights did not beat Stage
  15 (`best quick ablation visual ~0.9642`)

Interpretation: simple tuning is near saturation for the current pixel-MLP
setup. The remaining controlled error is mostly level `2` versus level `3`
boundary assignment, while WQY transfer remains dominated by source adaptation.

## Target 17: Explainable Level-2 Boundary Rules

Goal: understand whether target level `2` can be explained as a simple geometric
boundary inside the `ge2` source mask.

Command:

```powershell
python scripts/run_boundary_rules.py
```

Rule family:

- source: target-derived `ge2` mask
- inside source: predict level `3` if inside-neighbor count and edge-band tests
  pass; otherwise level `2`
- outside source: predict level `1` for simple right-down shadow candidates,
  otherwise level `0`
- ranking: CJK visual score, then ink F1, then shadow F1

Initial run:

- best rule: `n2_band0_plain`
- CJK visual score: `0.9361`
- CJK ink F1: `0.9355`
- CJK shadow F1: `0.8877`
- CJK foreground IoU: `0.9515`

Feature stats:

- CJK level-2 pixels: `16576`
- CJK level-3 pixels: `82378`
- level-2 distance to outside ge2 mask: `16567` at distance `1`, `9` at
  distance `2`
- level-3 distance to outside ge2 mask: `79454` at distance `1`, `2921` at
  distance `2`, `3` at distance `3`

Interpretation: level `2` is almost always a boundary pixel in the `ge2` mask,
but that condition is not sufficient because most level `3` pixels are also
near the boundary in 15x15 CJK glyphs. A simple neighbor-count rule is
explainable and decent, but the MLP's extra gain comes from richer local context.

## Target 18: Patch Model Readiness

Goal: decide whether the next model should use local patches by comparing where
the controlled `ge2` MLP beats the explainable boundary rule.

Command:

```powershell
python scripts/build_patch_readiness.py
```

Dataset:

- source mask: target-derived `ge2`, because it is the best controlled 1bpp
  source for the current objective
- split focus: CJK only
- included pixels: target level `2` and `3`
- patch: compact 9x9 binary source patch centered on each included pixel
- labels: target level, Stage 15 MLP prediction, Stage 17 rule prediction, and
  category

Output:

- `stage18_patch_readiness/patch_readiness_metadata.json`
- `stage18_patch_readiness/patch_readiness_patches.jsonl`
- `stage18_patch_readiness/patch_readiness_contact.png`

Initial run:

- CJK ge2 records: `98954`
- MLP level-2/3 pixel accuracy: `0.9380`
- boundary-rule level-2/3 pixel accuracy: `0.8905`
- `rule_wrong_mlp_right`: `6177`
- `mlp_wrong_rule_right`: `1474`
- `both_wrong`: `4660`

By target level:

- target level `2`: `5315` rule-wrong/MLP-right, `271` MLP-wrong/rule-right,
  `3489` both-wrong
- target level `3`: `862` rule-wrong/MLP-right, `1203` MLP-wrong/rule-right,
  `1171` both-wrong

Interpretation: the MLP's extra value is concentrated on subtle level-2 edge
pixels, which supports a patch-aware next stage. The contact sheet should be
read as four rows per example: source `ge2` patch, target patch, MLP patch, rule
patch.

## Target 19: Patch-Only Ge2 Classifier

Goal: test whether local 9x9 source patches alone can learn the `ge2` level
`2` versus level `3` split.

Command:

```powershell
python scripts/train_patch_classifier.py
```

Model family:

- `logistic_balanced`: linear classifier over 9x9 binary `ge2` patch bits
- `patch_mlp`: one-hidden-layer MLP over the same 81 patch bits

Scope:

- train/evaluate on CJK target-derived `ge2` pixels only
- labels are only level `2` and level `3`
- outside `ge2`, use fixed right-down shadow level `1`; otherwise level `0`

Initial run:

- training pixels: `98954`
- label counts: level `2` = `16576`, level `3` = `82378`
- best model: `patch_mlp`
- best CJK visual score: `0.9590`
- best CJK ge2 level-2/3 pixel accuracy: `0.9530`
- `logistic_balanced` CJK visual score: `0.8990`

Patch MLP confusion:

- level `2 -> 2`: `13619`
- level `2 -> 3`: `2957`
- level `3 -> 2`: `1693`
- level `3 -> 3`: `80685`

Interpretation: local patch context is enough to improve the core/edge split,
but it does not solve the whole style score because shadow placement is still a
fixed rule. Stage 20 should either model shadow as a separate `0/1` classifier
or combine the Stage 19 `2/3` classifier with a learned shadow head.

## Target 20: Learned Shadow Classifier

Goal: split the style problem into two learned patch heads: level `2/3` inside
the `ge2` source mask and level `0/1` outside it.

Command:

```powershell
python scripts/train_shadow_classifier.py
```

Model family:

- core/edge head: Stage19-style patch MLP over 9x9 `ge2` source patches
- shadow head: `shadow_logistic_balanced` and `shadow_patch_mlp`, both over the
  same 9x9 source patch centered on outside-`ge2` pixels

Initial run:

- core/edge pixels: `98954`
- shadow pixels: `244846`
- shadow labels: level `0` = `167129`, level `1` = `77717`
- best shadow model: `shadow_patch_mlp`
- CJK visual score: `0.9739`
- CJK ink F1: `0.9738`
- CJK shadow F1: `0.9557`
- CJK foreground IoU: `0.9768`
- shadow 0/1 accuracy: `0.9831`
- shadow 0/1 F1: `0.9735`

Shadow MLP confusion:

- level `0 -> 0`: `164561`
- level `0 -> 1`: `2568`
- level `1 -> 0`: `1582`
- level `1 -> 1`: `76135`

Interpretation: once shadow placement is learned separately, the target-derived
controlled baseline jumps past the earlier pixel MLPs. This makes Stage 20 the
current best style-learning reference. The remaining big unknown is transfer to
real WQY source masks, where source alignment and glyph-shape differences are
still expected to dominate.

## Target 21: External Source Transfer Evaluation

Goal: evaluate whether the Stage20 two-head style model transfers to real WQY
source masks, and report source-mask alignment next to style metrics.

Command:

```powershell
python scripts/eval_external_sources.py
```

Default external sources:

- `wqy13`: Stage 2 WQY Sharp size 13 source metadata
- `wqy14`: Stage 14 WQY Sharp size 14 source metadata

Initial CJK run:

- WQY13 source mask vs target ge2 F1: `0.4056`
- WQY13 source mask vs target ge2 IoU: `0.2666`
- WQY13 visual score: `0.4746`
- WQY13 ink F1: `0.3833`
- WQY13 shadow F1: `0.3458`
- WQY14 source mask vs target ge2 F1: `0.5576`
- WQY14 source mask vs target ge2 IoU: `0.4093`
- WQY14 visual score: `0.6034`
- WQY14 ink F1: `0.5437`
- WQY14 shadow F1: `0.4906`

Interpretation: Stage20 is a strong controlled style transform, but WQY transfer
is bottlenecked by the source mask. WQY14 is still materially closer than WQY13,
yet its source mask F1 against target `ge2` is only `0.5576`, so the next
productive work is source adaptation rather than adding style-model capacity.
