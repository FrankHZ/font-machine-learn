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

The test exports `a.NFTR` into a temporary directory and verifies the key source
font facts: `RTFN` mode, `15x15`, `2bpp`, `57` bytes per glyph, and `1814`
glyphs.

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

## Next Milestones

See `docs/targets.md` for the working target split.
See `docs/deliverables.md` for stage deliverables and commit checkpoints.

1. Train the next model on Stage 11 pairs: 1bpp visible masks as input and 2bpp
   target levels as labels.
2. Keep CJK as the primary split and non-CJK as a guard split.
3. Compare model outputs by contact sheet first, then by level-aware visual
   metrics.
