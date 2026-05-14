# Font Machine Learn

This project is an experiment for learning the visual style of a Nintendo DS
bitmap font: small 15 x 15 px glyphs with 2bpp levels for transparent,
shadow, edge transition, and main stroke pixels.

The core task is **not** to invent or correct glyph shapes. The intended
pipeline is:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

The model/rules should learn style layering:

- `3`: main stroke core
- `2`: edge/anti-alias transition around the core
- `1`: right-down shadow
- `0`: transparent/background

`wqy-zenhei.ttc` is the future real input font source and a useful validation
asset. The current `a.NFTR` target can also be quantized to 1bpp to create the
cleanest paired training data for learning the 1bpp-to-2bpp style transform.

The first milestones were intentionally small:

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
│   ├── extract_target_glyphs.py    # Split NFTR into per-glyph target dataset
│   ├── render_source_glyphs.py     # Render WQY Sharp source glyph dataset
│   ├── run_shadow_baseline.py      # Rule-based source-to-shadow baseline
│   ├── train_mlp_baseline.py       # Small trainable pixel-level MLP baseline
│   ├── tune_shadow_baseline.py      # Visual-score-oriented shadow rule search
│   ├── build_review_report.py       # Side-by-side baseline review artifacts
│   ├── run_binary_diagnostic.py     # Quantize target to 1bpp and score masks
│   ├── search_weight_baseline.py     # Search source weight/offset before shadow
│   ├── run_cjk_style_baseline.py     # CJK-focused fixed-style baseline
│   ├── run_cjk_edges_baseline.py     # CJK level-2 edge transition refinement
│   ├── build_1bpp_style_dataset.py   # Formal 1bpp-mask to 2bpp-style dataset
│   ├── compare_target_masks_to_source.py # Compare target >=2/==3 masks to WQY
│   ├── run_wqy_alignment_diagnostic.py # Search WQY alignment/weight by mask mode
│   ├── train_style_mlp.py        # Controlled 1bpp-to-2bpp style MLP
│   ├── run_boundary_rules.py     # Explainable ge2 level-2/3 boundary rules
│   ├── build_patch_readiness.py  # Stage 18 patch/error dataset for next model
│   ├── train_patch_classifier.py # Stage 19 patch-only ge2 2/3 classifiers
│   ├── train_shadow_classifier.py # Stage 20 learned 0/1 shadow classifier
│   ├── eval_external_sources.py  # Stage 21 WQY source-mask transfer eval
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
    └── processed/
        └── glyphs/                # Stage-sorted generated glyph artifacts
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

The default harness is intentionally fast. It verifies parser invariants and
small pure functions without rebuilding every historical stage.

Run full stage-export smoke tests only when changing stage exporters:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
python -m unittest discover
```

The slow tests export generated artifacts into `.tmp/tests/`, which is ignored
by Git.

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
- `1`: right-down shadow
- `2`: edge/anti-alias transition around the main stroke
- `3`: main stroke

The exporter also has a raw fallback for diagnostic work with incorrectly
extracted or compressed files:

```powershell
python scripts/export_nftr.py some.raw --cell-width 15 --cell-height 15 --bpp 2 --offset 0
```

## Glyph Output Layout

Generated glyph artifacts are grouped by stage under `data/processed/glyphs/`.
Keep the root of `glyphs/` for stage folders only; old flat outputs may be moved
under `legacy_flat/`.

```text
data/processed/glyphs/
├── stage1_target/          # target PNGs, metadata, contact sheet
├── stage2_source/          # WQY source PNGs and source/target contact sheet
├── stage3_shadow/          # first rule-based shadow baseline
├── stage4_mlp/             # trainable MLP baseline
├── stage5_tuned_shadow/    # visual-score shadow search
├── stage6_review/          # side-by-side human review artifacts
├── stage7_binary/          # 1bpp target diagnostic
├── stage8_weight/          # source weight/offset search
├── stage9_cjk_style/       # CJK-first fixed-style baseline
├── stage10_cjk_edges/      # CJK level-2 edge transition refinement
├── stage11_1bpp_style/     # formal 1bpp-mask to 2bpp-style dataset
├── stage12_target_masks/   # compare target >=2 and ==3 masks to WQY source
├── stage13_wqy_alignment/  # WQY offset/weight search by target mask mode
├── stage15_style_mlp/      # controlled target-derived style-learning baseline
├── stage16_style_mlp_tuning/ # small capacity/feature tuning over Stage 15
├── stage17_boundary_rules/ # explainable ge2 level-2/3 boundary search
├── stage18_patch_readiness/ # CJK ge2 patch/error index for next model choice
├── stage19_patch_classifier/ # patch-only ge2 level-2/3 classifier outputs
├── stage20_shadow_classifier/ # learned shadow + patch MLP combined output
├── stage21_external_eval/ # Stage20 two-head model on WQY source masks
└── legacy_flat/            # archived outputs from the old flat layout
```

When adding a new stage, add a `stageN_name/` directory and put that stage's
PNG directories, metadata, search reports, and contact sheets inside it.

## Extract Target Glyphs

Run:

```powershell
python scripts/extract_target_glyphs.py a.NFTR
```

The command writes disposable dataset artifacts under
`data/processed/glyphs/stage1_target/`:

- `target/*.png`
- `target_metadata.json`
- `target_contact.png`

The metadata is the important contract for later stages: glyph index, Shift-JIS
codes, decoded characters where possible, width metrics, and per-level 2bpp
histograms.

## Render WQY Source Glyphs

Run:

```powershell
python scripts/render_source_glyphs.py
```

Default settings match the previous visual research:

- font: `wqy-zenhei.ttc`
- face: `WenQuanYi Zen Hei Sharp`, TTC index `2`
- size: `13`
- cell: `15x15`

The command writes disposable paired-source artifacts under
`data/processed/glyphs/stage2_source/`:

- `source/*.png`
- `source_metadata.json`
- `source_target_contact.png`

Most rendered CJK ink boxes should land around `12-13px` wide, which keeps them
close to the original NFTR cells. WQY is a candidate production input font, but
it is not the only or preferred training source. For style learning, the
strongest paired dataset comes from target glyphs quantized to 1bpp as input and
the original target 2bpp glyphs as labels.

## Run the Shadow Baseline

Run:

```powershell
python scripts/run_shadow_baseline.py
```

This creates a simple, explainable baseline from the WQY source glyphs:

- source ink becomes level `3`;
- right and down neighbors become level `2`;
- down-right shadow becomes level `1`.

The command writes:

- `baseline_shadow/*.png`
- `baseline_shadow_metadata.json`
- `baseline_shadow_contact.png`

The contact sheet stacks source, baseline prediction, and target glyphs. Metadata
includes pixel accuracy, mean absolute error, and foreground IoU per glyph plus
dataset means.

## Train the MLP Baseline

Run:

```powershell
python scripts/train_mlp_baseline.py
```

This keeps the dependency stack light by using `scikit-learn` instead of a heavy
deep-learning framework. It trains a small pixel-level MLP classifier:

- features: `3x3` source patch, normalized pixel coordinates, and edge distance
  hints;
- label: target 2bpp level `0..3`;
- prediction: full 1814-glyph baseline output.

The command writes:

- `baseline_mlp/*.png`
- `baseline_mlp_metadata.json`
- `baseline_mlp_contact.png`

## Tune Visual Shadow Rules

Run:

```powershell
python scripts/tune_shadow_baseline.py
```

The MLP baseline is useful as a training harness, but plain pixel accuracy can
reward bland background predictions. The tuned shadow baseline scores rules with
more visual terms:

- level-3 ink F1
- level-1/2 shadow F1
- foreground IoU
- target-weighted level similarity
- isolated foreground noise

The command writes:

- `baseline_tuned_shadow/*.png`
- `baseline_tuned_shadow_metadata.json`
- `baseline_tuned_shadow_search.json`
- `baseline_tuned_shadow_contact.png`

## Build Review Artifacts

Run:

```powershell
python scripts/build_review_report.py
```

The review sheet stacks each selected glyph horizontally in this order:
source, rule shadow, tuned shadow, MLP, target. The selected rows are the worst
cases by visual score, so they are useful for human inspection before changing
the next model.

The command writes:

- `review_report.json`
- `review_worst_cases.json`
- `review_contact.png`

## Run 1bpp Diagnostics

Run:

```powershell
python scripts/run_binary_diagnostic.py
```

This quantizes target glyphs to foreground/background and scores baselines as
binary masks. It also provides the canonical paired input for the main learning
task: target-derived 1bpp masks as source, original target 2bpp glyphs as label.

The command writes:

- `target_1bpp/*.png`
- `binary_diagnostic_metadata.json`
- `binary_diagnostic_contact.png`

## Search WQY Source Weight

Run:

```powershell
python scripts/search_weight_baseline.py
```

This searches small WQY source-glyph offsets and dilation kernels against the
1bpp target mask, then applies the tuned shadow rule to the best weighted source.
These stages are diagnostic for using WQY as a real input font; they should not
replace the main 1bpp-to-2bpp style-learning objective.

The command writes:

- `source_weighted/*.png`
- `baseline_weighted_shadow/*.png`
- `weight_search.json`
- `weight_search_metadata.json`
- `baseline_weighted_shadow_contact.png`

## Run the CJK Style Baseline

Run:

```powershell
python scripts/run_cjk_style_baseline.py
```

This stage focuses ranking on CJK glyphs. The style rule is constrained:

- source core pixels are level `3`
- source-adjacent edge transition pixels are level `2`
- right-down `(1, 1)` shadow pixels are level `1`

The command writes:

- `stage9_cjk_style/baseline_cjk_style/*.png`
- `stage9_cjk_style/cjk_style_metadata.json`
- `stage9_cjk_style/cjk_style_search.json`
- `stage9_cjk_style/cjk_style_contact.png`
- `stage9_cjk_style/cjk_style_worst_contact.png`

## Refine CJK Edge Transitions

Run:

```powershell
python scripts/run_cjk_edges_baseline.py
```

This keeps level `3` as the source-derived core and level `1` as the fixed
right-down shadow, then searches where level `2` edge/anti-alias pixels may be
placed around the core. Candidate edge pixels are constrained by nearby core
neighbor count so dense CJK glyphs do not get filled in.

The command writes:

- `stage10_cjk_edges/baseline_cjk_edges/*.png`
- `stage10_cjk_edges/cjk_edges_metadata.json`
- `stage10_cjk_edges/cjk_edges_search.json`
- `stage10_cjk_edges/cjk_edges_contact.png`
- `stage10_cjk_edges/cjk_edges_worst_contact.png`

## Build the 1bpp Style Dataset

Run:

```powershell
python scripts/build_1bpp_style_dataset.py
```

This is the formal training direction. It uses target-derived 1bpp masks as the
input side and the original NFTR 2bpp glyphs as labels. Later, any complete 1bpp
font, including WQY Sharp, should be able to pass through the same style
pipeline.

Important: the 1bpp input is the visible glyph silhouette, not a pre-labeled
level-3 core. The Stage 11 baseline keeps that silhouette fixed, then searches a
simple decomposition into level `3` core, level `2` edge/transition pixels, and
level `1` right-down shadow pixels.

Outputs:

- `stage11_1bpp_style/input_1bpp/*.png`
- `stage11_1bpp_style/baseline_2bpp/*.png`
- `stage11_1bpp_style/style_pairs_metadata.json`
- `stage11_1bpp_style/style_baseline_search.json`
- `stage11_1bpp_style/style_baseline_contact.png`
- `stage11_1bpp_style/style_baseline_worst_contact.png`

Current baseline:

- best rule: `core_n3_right_down_dx-1_dy-1`
- glyphs: `1814`, CJK glyphs: `1528`
- CJK foreground F1/IoU: `1.0000` / `1.0000`
- CJK visual score: `0.8170`
- all visual score: `0.8257`
- non-CJK visual score: `0.8718`

## Compare Target 1bpp Mask Choices

Run:

```powershell
python scripts/compare_target_masks_to_source.py
```

This compares two target-derived 1bpp masks against the WQY Sharp source glyphs:

- `ge2`: target level `>= 2`, meaning main stroke plus edge/transition pixels
- `eq3`: target level `== 3`, meaning main stroke core only

The contact sheet stacks each glyph vertically as WQY source, `ge2`, `eq3`, and
original 2bpp target.

Current result:

- CJK `ge2` F1/IoU: `0.4141` / `0.2657`
- CJK `eq3` F1/IoU: `0.3903` / `0.2517`
- CJK source-nonempty winner: `ge2`

Interpretation: target `>=2` is slightly closer to WQY Sharp than `==3`, mostly
because WQY Sharp is a little heavier than the NFTR level-3 core. Both scores
remain low because WQY and NFTR differ in glyph shape, placement, and stroke
weight. This does not make either binary mask a perfect training source:
quantizing level `2` upward or downward discards the anti-alias/edge role that
gives the target font its look.

## Diagnose WQY Alignment and Weight

Run:

```powershell
python scripts/run_wqy_alignment_diagnostic.py
```

This searches small WQY source offsets and dilation kernels separately against
three target mask views:

- `visible`: target level `> 0`
- `ge2`: target level `>= 2`
- `eq3`: target level `== 3`

The contact sheet stacks each glyph as WQY source, aligned/target `visible`,
aligned/target `ge2`, aligned/target `eq3`, and original 2bpp target.

Current CJK source-nonempty result:

- `visible`: best `cardinal_dx+0_dy+1`, F1/IoU `0.8282` / `0.7111`
- `ge2`: best `right_down_dx-1_dy+0`, F1/IoU `0.6388` / `0.4674`
- `eq3`: best `right_down_dx-1_dy+0`, F1/IoU `0.5861` / `0.4232`

Interpretation: WQY can be made fairly close to the full visible NFTR
silhouette only by adding weight. It remains much less aligned with `>=2` and
`==3`, so WQY adaptation should be separated from 2bpp style-layer learning.

For tiny bitmap fonts, do not treat dilation/boldening as a normal production
fix. A better first check is the native bitmap strike size. Rendering WQY Sharp
at size `14` in the same `15x15` cell improved raw CJK source alignment without
synthetic thickening:

- size `13` raw CJK `>=2` F1/IoU: `0.4141` / `0.2657`
- size `13` raw CJK `==3` F1/IoU: `0.3903` / `0.2517`
- size `14` raw CJK `>=2` F1/IoU: `0.5590` / `0.4078`
- size `14` raw CJK `==3` F1/IoU: `0.5326` / `0.3815`

The size-14 diagnostic output is under `stage14_wqy_size14/`.

## Train the Controlled Style MLP

Run:

```powershell
python scripts/train_style_mlp.py
```

This is the first trainable baseline for the corrected task. It trains the same
small pixel-level MLP on three target-derived 1bpp inputs:

- `visible`: target level `> 0`
- `ge2`: target level `>= 2`
- `eq3`: target level `== 3`

Labels are always the original NFTR 2bpp levels `0..3`. WQY 13/14 are evaluated
as external inputs only and do not choose the best controlled mode.

Current CJK result:

- controlled `visible` visual score: `0.9031`
- controlled `ge2` visual score: `0.9651`
- controlled `eq3` visual score: `0.9633`
- external WQY13 visual score: `0.4707`
- external WQY14 visual score: `0.6029`

Interpretation: controlled 1bpp-to-2bpp style learning works. WQY14 transfers
better than WQY13, but both remain source-adaptation problems rather than style
learning failures.

## Tune the Controlled Style MLP

Run:

```powershell
python scripts/train_style_mlp.py --hidden-units 128 --max-iter 100 --random-seed 17 --out-dir data/processed/glyphs/stage16_style_mlp_tuning --metadata data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_metadata.json --contact-sheet data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_contact.png
```

This keeps the Stage 15 model family and only checks whether extra capacity can
improve the `ge2` controlled result. A separate edge-feature/class-weight
prototype did not beat Stage 15, so it was kept as an ablation result rather
than promoted into the main code.

Current CJK result:

- Stage 15 `ge2` visual score: `0.9651`
- Stage 16 tuned `ge2` visual score: `0.9654`
- Stage 16 tuned `eq3` visual score: `0.9649`
- Stage 16 WQY13 external visual score: `0.4677`
- Stage 16 WQY14 external visual score: `0.5995`

Interpretation: this is a marginal controlled improvement, not a new
breakthrough. Remaining gains likely need a better formulation for level `2`
edge pixels or source adaptation, not just more MLP capacity.

## Explain Level-2 Boundaries

Run:

```powershell
python scripts/run_boundary_rules.py
```

This searches simple rules for the main remaining controlled error: deciding
whether a `ge2` source pixel should be target level `2` or `3`. The best rule is
selected by CJK visual score.

Current result:

- best rule: `n2_band0_plain`
- CJK visual score: `0.9361`
- CJK ink F1: `0.9355`
- CJK shadow F1: `0.8877`

Interpretation: a simple rule explains much of the target style, but it remains
well below Stage 15/16 MLP scores. Level `2` is mostly a ge2 boundary pixel:
`16567 / 16576` CJK level-2 pixels are distance `1` from outside the `ge2` mask.
However, many level-3 pixels are also near that boundary, so the remaining
decision needs richer local context than a single neighbor-count rule.

## Build Patch Readiness Dataset

Run:

```powershell
python scripts/build_patch_readiness.py
```

This does not train a CNN yet. It compares the Stage 15 `ge2` MLP output against
the Stage 17 explainable boundary rule and writes a CJK-focused pixel patch
index:

- `data/processed/glyphs/stage18_patch_readiness/patch_readiness_metadata.json`
- `data/processed/glyphs/stage18_patch_readiness/patch_readiness_patches.jsonl`
- `data/processed/glyphs/stage18_patch_readiness/patch_readiness_contact.png`

Each JSONL row is a CJK target level `2` or `3` pixel with a compact 9x9
`ge2` source patch, target label, MLP label, rule label, and error category.

Current result:

- CJK ge2 patch records: `98954`
- MLP ge2 2/3 pixel accuracy: `0.9380`
- boundary-rule ge2 2/3 pixel accuracy: `0.8905`
- `rule_wrong_mlp_right`: `6177`
- `mlp_wrong_rule_right`: `1474`
- `both_wrong`: `4660`

Interpretation: the MLP's improvement over the explainable rule is real and is
mostly on level-2 edge pixels, so the next useful model should be patch-aware
rather than just another scalar boundary rule.

## Train Patch Classifier

Run:

```powershell
python scripts/train_patch_classifier.py
```

This trains two CJK-focused classifiers for the `ge2` source mask:

- `logistic_balanced`: a linear 9x9 patch-only baseline
- `patch_mlp`: a small MLP over the same 9x9 patch bits

Both models only decide whether a `ge2` source pixel is target level `2` or
`3`. Pixels outside `ge2` still use the fixed right-down shadow rule.

Current result:

- best model: `patch_mlp`
- CJK visual score: `0.9590`
- CJK ge2 2/3 pixel accuracy: `0.9530`
- `patch_mlp` confusion: level `2` correct `13619`, level `2 -> 3` `2957`;
  level `3 -> 2` `1693`, level `3` correct `80685`
- `logistic_balanced` CJK visual score: `0.8990`

Interpretation: patch MLP improves the core/edge split beyond Stage 15's ge2
2/3 accuracy, but the full visual score is lower than Stage 15 because Stage 19
keeps shadow as a fixed rule. The next split should treat `2/3` and `0/1`
shadow placement as separate subproblems.

## Train Shadow Classifier

Run:

```powershell
python scripts/train_shadow_classifier.py
```

This stage combines two learned patch heads over the target-derived `ge2` mask:

- a Stage19-style MLP for level `2` versus `3` inside `ge2`
- a shadow classifier for level `0` versus `1` outside `ge2`

Current result:

- best shadow model: `shadow_patch_mlp`
- CJK visual score: `0.9739`
- CJK ink F1: `0.9738`
- CJK shadow F1: `0.9557`
- CJK foreground IoU: `0.9768`
- shadow 0/1 accuracy: `0.9831`
- shadow F1 from 0/1 confusion: `0.9735`

Interpretation: learned shadow placement fixes the main Stage19 weakness and
beats the earlier controlled MLP baselines. The next useful work is to package
this as the current controlled best baseline, then test how it behaves on real
WQY-derived source masks.

## Evaluate External Sources

Run:

```powershell
python scripts/eval_external_sources.py
```

This retrains the Stage20 two-head patch model on target-derived `ge2` masks,
then evaluates it on rendered WQY source masks. By default it uses:

- `wqy13`: `data/processed/glyphs/stage2_source/source_metadata.json`
- `wqy14`: `data/processed/glyphs/stage14_wqy_size14/source_metadata.json`
- `song13`: `data/processed/glyphs/stage21_external_eval_sources/song13/source_metadata.json`, if present
- `song14`: `data/processed/glyphs/stage21_external_eval_sources/song14/source_metadata.json`, if present

Current CJK result:

- WQY13 source mask vs target ge2 F1: `0.4056`
- WQY13 visual score after style model: `0.4746`
- WQY14 source mask vs target ge2 F1: `0.5576`
- WQY14 visual score after style model: `0.6034`
- Song13 source mask vs target ge2 F1: `0.4258`
- Song13 visual score after style model: `0.4696`
- Song14 source mask vs target ge2 F1: `0.4211`
- Song14 visual score after style model: `0.4768`

Interpretation: transfer is dominated by source mask quality. Stage20 has a
strong controlled style transform, but WQY source masks do not yet align well
enough with target `ge2` structure for the style model to shine. The bitmap Song
faces render cleanly, but their source masks are not closer than WQY14 for this
target.

## Next Milestones

See `docs/targets.md` for the working target split.
See `docs/deliverables.md` for stage deliverables and commit checkpoints.

1. Keep Stage 20 as the current controlled best baseline.
2. Work on source adaptation for WQY masks before increasing style-model
   capacity.
3. Keep CJK as the primary split and non-CJK as a guard split.
4. Compare model outputs by contact sheet first, then by level-aware visual
   metrics.
