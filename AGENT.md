# Agent Notes

This repository is for experimenting with machine learning over tiny bitmap
fonts. The target style is a 15x15, 2bpp Nintendo DS font with separate
transparent, right-down shadow, edge-transition, and main-stroke levels.

The core project goal is:

```text
complete 1bpp bitmap glyph set -> NFTR-style 2bpp layered glyphs
```

Do not frame the main task as glyph-shape correction. WQY Sharp is a future real
input font and a useful validation source, but the central learning problem is
style layering from a 1bpp mask into 2bpp levels.

## Working Style

- Keep the first pipeline boring and inspectable: source file in, PNG/JSON out.
- Prefer small Python modules under `src/font_machine_learn/`.
- Keep scripts thin; put reusable logic in the package.
- The current `a.NFTR` is the decompressed source font. It uses reversed Nitro
  section tags: `RTFN`, `FNIF`, `PLGC`, `HDWC`, `PAMC`.
- Expected source-font fields: `15x15`, `2bpp`, `57` bytes per glyph,
  baseline `15`, glyph count `1814`.
- When a format detail is guessed, expose it as a CLI option and write the
  chosen value into JSON metadata.

## Important Files

- `a.NFTR`: decompressed target/source-style NFTR.
- `fonts/wqy-zenhei.ttc`: source bitmap font family; WQY Sharp face index `2` is a
  future real input font for the 1bpp-to-2bpp style pipeline.
- `docs/targets.md`: target split and expected outputs.
- `docs/deliverables.md`: stage deliverables, verification, and commit rhythm.
- `scripts/export_nftr.py`: main smoke-test command.
- `scripts/extract_target_glyphs.py`: Stage 1 target glyph dataset command.
- `scripts/render_source_glyphs.py`: Stage 2 WQY Sharp source glyph command.
- `scripts/run_shadow_baseline.py`: Stage 3 rule-based transform command.
- `scripts/train_mlp_baseline.py`: Stage 4 trainable pixel-level MLP baseline.
- `scripts/tune_shadow_baseline.py`: Stage 5 visual shadow rule search.
- `scripts/build_review_report.py`: Stage 6 side-by-side review artifacts.
- `scripts/run_binary_diagnostic.py`: Stage 7 1bpp target diagnostic.
- `scripts/search_weight_baseline.py`: Stage 8 source weight/offset search.
- `scripts/run_cjk_style_baseline.py`: Stage 9 CJK-focused fixed-style baseline.
- `scripts/run_cjk_edges_baseline.py`: Stage 10 CJK level-2 edge refinement.
- `scripts/build_1bpp_style_dataset.py`: Stage 11 formal 1bpp-to-2bpp dataset.
- `scripts/compare_target_masks_to_source.py`: Stage 12 target mask comparison.
- `scripts/run_wqy_alignment_diagnostic.py`: Stage 13 WQY alignment/weight search.
- `scripts/train_style_mlp.py`: Stage 15 controlled 1bpp-to-2bpp style MLP.
- `scripts/run_boundary_rules.py`: Stage 17 explainable ge2 boundary rules.
- `scripts/build_patch_readiness.py`: Stage 18 patch/error dataset for the next
  model decision.
- `scripts/train_patch_classifier.py`: Stage 19 patch-only ge2 level-2/3
  classifiers.
- `scripts/train_shadow_classifier.py`: Stage 20 learned 0/1 shadow classifier
  combined with the ge2 patch MLP.
- `scripts/eval_external_sources.py`: Stage 21 transfer evaluation on the
  current Song13 source baseline; pass explicit `--source` values for old
  comparisons.
- `scripts/train_song13_adapter.py`: Stage 22 source-mask adapter before the
  Stage20 two-head style model.
- `scripts/train_song13_calibrated.py`: Stage 23 threshold sweep for the Song13
  adapter, ranked with an ink-ratio/readability penalty.
- `scripts/train_song13_add_only.py`: Stage 24 source-preserving adapter that
  can only add ge2 pixels outside the Song13 source mask.
- `scripts/run_song13_source_locked.py`: Stage 25 source-locked style rule
  sweep; source pixels are never removed and only receive level `2`/`3`.
- `scripts/train_song13_layer_mlp.py`: Stage 26 source-locked learned layer
  assignment; target `ge2` trains layer semantics, Song13 shape stays fixed.
- `scripts/check_env.py`: local dependency sanity check.
- `src/font_machine_learn/nftr.py`: parser/exporter implementation.
- `tests/test_nftr_export.py`: fixed smoke-test harness.
- `data/processed/`: default place for generated PNG/JSON outputs.
- `src/font_machine_learn/paths.py`: canonical generated-output paths.

## Glyph Output Layout

Generated glyph artifacts must be grouped by stage under
`data/processed/glyphs/`.

- `stage1_target/`: NFTR target glyphs, metadata, contact sheet.
- `stage2_source/`: WQY source glyphs and source/target contact sheet.
- `stage3_shadow/`: first rule-based shadow baseline.
- `stage4_mlp/`: trainable MLP baseline.
- `stage5_tuned_shadow/`: visual-score shadow search.
- `stage6_review/`: side-by-side human review artifacts.
- `stage7_binary/`: 1bpp target diagnostic.
- `stage8_weight/`: WQY source weight/offset diagnostic.
- `stage9_cjk_style/`: CJK-focused style baseline over WQY-shaped input.
- `stage10_cjk_edges/`: CJK-focused level-2 edge transition refinement over
  WQY-shaped input.
- `stage11_1bpp_style/`: target-derived 1bpp masks paired with original 2bpp
  labels; this is the main training direction.
- `stage12_target_masks/`: compare target `>=2` and `==3` 1bpp masks against
  WQY source glyphs.
- `stage13_wqy_alignment/`: search WQY source offset/weight against target
  `visible`, `>=2`, and `==3` mask views.
- `stage14_wqy_size14/`: WQY Sharp size-14 raw source diagnostic; use this to
  compare native bitmap strike size before considering any synthetic thickening.
- `stage15_style_mlp/`: controlled target-derived style-learning MLP outputs.
- `stage16_style_mlp_tuning/`: capacity/ablation run over Stage 15; current
  tuned result is only marginally better than Stage 15.
- `stage17_boundary_rules/`: explainable rule search for level `2` vs `3`
  inside `ge2` target-derived source.
- `stage18_patch_readiness/`: CJK `ge2` 9x9 patch/error index comparing Stage
  15 MLP and Stage 17 rule outputs.
- `stage19_patch_classifier/`: patch-only classifiers for assigning level `2`
  vs `3` inside the `ge2` source mask.
- `stage20_shadow_classifier/`: separate learned shadow head combined with the
  Stage19-style patch MLP; current controlled best baseline.
- `stage21_external_eval/`: Stage20 two-head model evaluated on the current
  external source baseline. The default is now Song 13px only; use explicit
  `--source` arguments for historical WQY/Song12 comparisons.
- `stage21_external_eval_sources/`: optional rendered source datasets for
  external fonts. Current baseline source is WenQuanYi Bitmap Song 13px:
  `--font-size 15 --font-mode L --threshold 96 --x-offset -1 --y-offset 1`.
  This is a Song/Ming-style face with serifs, not a Gothic/Hei face; keep that
  as the baseline and avoid broad font comparisons unless explicitly requested.
- `stage22_song13_adapter/`: current source-adaptation experiment. It learns a
  Song13 source mask to target `ge2` adapter, then feeds the adapted mask into
  the Stage20 core/edge and shadow heads.
- `stage23_song13_calibrated/`: calibrated threshold sweep over the Stage22
  adapter family. Use this when Stage22's raw F1 improvement looks too dark or
  over-connected by eye.
- `stage24_song13_add_only/`: add-only source adapter. It preserves every
  Song13 source pixel and only learns outside-source additions.
- `stage25_song13_source_locked/`: current source-preserving baseline for
  Song13 style assignment. It does not adapt shape; it only assigns `2/3` inside
  source and `1` shadow outside source.
- `stage26_song13_layer_mlp/`: learned source-locked Song13 layer assignment.
  It trains core/edge and shadow heads on target-derived `ge2` masks, then
  applies them to Song13 without deleting source pixels.
- `legacy_flat/`: archived outputs from the old flat layout.

Do not add new generated PNG/JSON artifacts directly under the glyph root.
For a new stage, add constants to `src/font_machine_learn/paths.py`, make script
defaults use those constants, and document the stage directory here and in
README.

Level semantics for style work:

- `3`: main stroke core inferred inside the 1bpp visible silhouette.
- `2`: edge/anti-alias transition around the core, not generic stroke
  thickening.
- `1`: fixed right-down shadow unless a later stage explicitly changes the
  style constraint.
- `0`: transparent/background.

For Stage 11 and later, remember that the source 1bpp mask is a visible
silhouette containing core, edge, and shadow pixels together. Do not treat every
source pixel as level `3`; the task is to split that silhouette into 2bpp style
layers.

For tiny bitmap fonts, do not use dilation/boldening as a default adaptation
strategy. Check native strike size and placement first. WQY Sharp size `14` is a
current diagnostic candidate in the `15x15` cell; size `13` was the earlier
baseline.

Current external-source direction:

- Use WenQuanYi Bitmap Song 13px as the default baseline source.
- Render it as `font-size 15`, `font-mode L`, `threshold 96`, `x-offset -1`,
  `y-offset +1`.
- Treat serif feet as expected source shape, not a rendering error.
- Stage 22 improved CJK raw Song13 ge2 F1 from `0.5953` to adapted F1 `0.6844`
  and final visual score from `0.6320` to `0.6809`.
- Stage 23 calibrated the adapter threshold to `0.55`; adapted foreground ratio
  moved from Stage22's `0.3324` toward target `0.2878`, landing at `0.2994`.
- Stage22 deleted about `10.2%` of Song13 CJK source pixels; Stage23 deleted
  about `14.9%`. This means those adapters were learning target-shape
  replacement, not just style prep.
- Stage24 forbids deletion. Its best CJK source deleted ratio is `0.0000`, with
  adapted foreground ratio `0.2805` against target `0.2878`. Prefer this
  source-preserving direction before larger style heads.
- Stage25 removes shape adaptation entirely. Best rule
  `edge_n1_diag_plus_right_from_source` has source deleted ratio `0.0000`,
  CJK visual `0.6409`, and source `2/3` split `0.1220` / `0.8780`.
- Stage26 is the next direction: learn `2/3` and `0/1` layer assignment under
  the Stage25 source-lock contract. Do not train it to predict a replacement
  Song13 `ge2` mask. Current best candidate
  `core_patch_mlp_shadow_logistic_balanced_c055_s045` keeps source deletion at
  `0.0000`, but CJK visual `0.6339` is below Stage25's `0.6409`; treat it as a
  working learned harness, not a quality win. The script supports
  `--jobs N` for CPU-threaded candidate scoring and caches probability grids;
  prefer `--jobs 4` for local full runs. It also supports multi-source eval
  after one shared training pass: use `--eval-target-quantized
  --eval-source-jobs 3` to evaluate Song13, target `>=2`, and target `==3`.
  Current multi-source CJK visual scores are Song13 `0.6339`, target_ge2
  `0.9742`, and target_eq3 `0.8903`. For quick iteration, add
  `--eval-limit 256 --search-limit 128`; this limits final eval/export and
  brings the three-source command to about `29s`.

## Target Split

- Target 0: export `a.NFTR` to atlas PNG/JSON and keep the parser stable.
- Target 1: split NFTR glyphs into labeled per-glyph target records using
  `PAMC`, `HDWC`, and glyph indexes.
- Target 2: render matching source glyphs from `fonts/wqy-zenhei.ttc`, face index `2`,
  into `15x15` cells for validation against the future real input font.
- Target 3 and later early baselines explored WQY-shaped inputs and visual
  metrics. Treat those as diagnostics, not as the final learning formulation.
- Target 11 pivots the main dataset to target-derived `1bpp` visible masks as
  input and original NFTR `2bpp` glyphs as labels.

## Smoke Test

Run this after normal changes:

```powershell
python -m unittest discover
python scripts/check_env.py
```

Default tests should stay fast. Historical stage exporters are slow and are
skipped unless `FML_RUN_SLOW_TESTS=1` is set.

Run this only when changing stage exporters:

```powershell
$env:FML_RUN_SLOW_TESTS = "1"
python scripts/export_nftr.py a.NFTR --out data/processed/a_atlas.png
python scripts/extract_target_glyphs.py a.NFTR
python scripts/render_source_glyphs.py
python scripts/run_shadow_baseline.py
python scripts/train_mlp_baseline.py
python scripts/tune_shadow_baseline.py
python scripts/build_review_report.py
python scripts/run_binary_diagnostic.py
python scripts/search_weight_baseline.py
python scripts/run_cjk_style_baseline.py
python scripts/run_cjk_edges_baseline.py
python scripts/build_1bpp_style_dataset.py
python scripts/compare_target_masks_to_source.py
python scripts/run_wqy_alignment_diagnostic.py
python scripts/train_style_mlp.py
python scripts/train_style_mlp.py --hidden-units 128 --max-iter 100 --random-seed 17 --out-dir data/processed/glyphs/stage16_style_mlp_tuning --metadata data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_metadata.json --contact-sheet data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_contact.png
python scripts/run_boundary_rules.py
python scripts/build_patch_readiness.py
python scripts/train_patch_classifier.py
python scripts/train_shadow_classifier.py
python scripts/eval_external_sources.py
python scripts/train_song13_adapter.py
python scripts/train_song13_calibrated.py
python scripts/train_song13_add_only.py
python scripts/run_song13_source_locked.py
python scripts/train_song13_layer_mlp.py
python -m unittest discover
```

For faster slow-smoke verification, prefer the parallel runner:

```powershell
python scripts/run_slow_smokes.py --jobs 4
python scripts/run_slow_smokes.py --jobs 2 --pattern song13
```

It runs each `@slow_test` method in a separate `python -B -m unittest` process
with `FML_RUN_SLOW_TESTS=1` and `PYTHONDONTWRITEBYTECODE=1`.
The `song13` subset has been checked with `--jobs 3`: five slow smoke tests
completed in about `79s`, compared with about `189s` summed individual test
time.

Expected result:

- command exits successfully;
- `data/processed/a_atlas.png` exists;
- `data/processed/a_atlas.json` exists and records the detected/export mode.
- `data/processed/glyphs/stage1_target/target_metadata.json` exists after Stage 1.
- `data/processed/glyphs/stage2_source/source_metadata.json` exists after Stage 2.
- `data/processed/glyphs/stage3_shadow/baseline_shadow_metadata.json` exists after Stage 3.
- `data/processed/glyphs/stage4_mlp/baseline_mlp_metadata.json` exists after Stage 4.
- `data/processed/glyphs/stage5_tuned_shadow/baseline_tuned_shadow_metadata.json` exists after Stage 5.
- `data/processed/glyphs/stage6_review/review_report.json` exists after Stage 6.
- `data/processed/glyphs/stage7_binary/binary_diagnostic_metadata.json` exists after Stage 7.
- `data/processed/glyphs/stage8_weight/weight_search_metadata.json` exists after Stage 8.
- `data/processed/glyphs/stage9_cjk_style/cjk_style_metadata.json` exists after Stage 9.
- `data/processed/glyphs/stage10_cjk_edges/cjk_edges_metadata.json` exists after Stage 10.
- `data/processed/glyphs/stage11_1bpp_style/style_pairs_metadata.json` exists after Stage 11.
- `data/processed/glyphs/stage12_target_masks/target_mask_compare_metadata.json` exists after Stage 12.
- `data/processed/glyphs/stage13_wqy_alignment/wqy_alignment_metadata.json` exists after Stage 13.
- `data/processed/glyphs/stage15_style_mlp/style_mlp_metadata.json` exists after Stage 15.
- `data/processed/glyphs/stage16_style_mlp_tuning/style_mlp_tuning_metadata.json` exists after Stage 16.
- `data/processed/glyphs/stage17_boundary_rules/boundary_rule_metadata.json` exists after Stage 17.
- `data/processed/glyphs/stage18_patch_readiness/patch_readiness_metadata.json` exists after Stage 18.
- `data/processed/glyphs/stage19_patch_classifier/patch_classifier_metadata.json` exists after Stage 19.
- `data/processed/glyphs/stage20_shadow_classifier/shadow_classifier_metadata.json` exists after Stage 20.
- `data/processed/glyphs/stage21_external_eval/external_eval_metadata.json` exists after Stage 21.
- `data/processed/glyphs/stage22_song13_adapter/song13_adapter_metadata.json` exists after Stage 22.
- `data/processed/glyphs/stage23_song13_calibrated/song13_calibrated_metadata.json` exists after Stage 23.
- `data/processed/glyphs/stage24_song13_add_only/song13_add_only_metadata.json` exists after Stage 24.
- `data/processed/glyphs/stage25_song13_source_locked/song13_source_locked_metadata.json` exists after Stage 25.
- `data/processed/glyphs/stage26_song13_layer_mlp/song13_layer_mlp_metadata.json` exists after Stage 26.
- default unittest passes quickly and confirms the source NFTR shape.
- slow unittest passes when `FML_RUN_SLOW_TESTS=1` is explicitly enabled.

## Documentation Rules

- Update `docs/targets.md` when changing target boundaries or dataset outputs.
- Update `docs/deliverables.md` when stage deliverables or verification changes.
- Update README commands when the runnable harness changes.
- Record guessed format details as metadata fields and CLI options.
- Keep Stage 11+ wording focused on `1bpp visible mask -> 2bpp style layers`;
  do not describe it as WQY glyph-shape correction.

## Design Constraints

- Treat generated artifacts under `data/processed/` as disposable.
- Keep original font assets untouched.
- Keep heavy deep-learning frameworks out of the default install until the
  extraction path is understood.
- Use Pillow for image IO unless the project grows past its needs.
