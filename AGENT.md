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
- `wqy-zenhei.ttc`: source bitmap font family; WQY Sharp face index `2` is a
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
- planned `stage11_1bpp_style/`: target-derived 1bpp masks paired with original
  2bpp labels; this is the main training direction.
- `legacy_flat/`: archived outputs from the old flat layout.

Do not add new generated PNG/JSON artifacts directly under the glyph root.
For a new stage, add constants to `src/font_machine_learn/paths.py`, make script
defaults use those constants, and document the stage directory here and in
README.

Level semantics for style work:

- `3`: 1bpp-mask-derived main stroke core.
- `2`: edge/anti-alias transition around the core, not generic stroke
  thickening.
- `1`: fixed right-down shadow unless a later stage explicitly changes the
  style constraint.
- `0`: transparent/background.

## Target Split

- Target 0: export `a.NFTR` to atlas PNG/JSON and keep the parser stable.
- Target 1: split NFTR glyphs into labeled per-glyph target records using
  `PAMC`, `HDWC`, and glyph indexes.
- Target 2: render matching source glyphs from `wqy-zenhei.ttc`, face index `2`,
  into `15x15` cells for validation against the future real input font.
- Target 3 and later early baselines explored WQY-shaped inputs and visual
  metrics. Treat those as diagnostics, not as the final learning formulation.
- Target 11 should pivot the main dataset to target-derived `1bpp` masks as
  input and original NFTR `2bpp` glyphs as labels.

## Smoke Test

Run this after changes:

```powershell
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
python -m unittest discover
python scripts/check_env.py
```

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
- planned `data/processed/glyphs/stage11_1bpp_style/style_pairs_metadata.json`
  should exist after Stage 11.
- unittest passes and confirms the source NFTR shape.

## Documentation Rules

- Update `docs/targets.md` when changing target boundaries or dataset outputs.
- Update `docs/deliverables.md` when stage deliverables or verification changes.
- Update README commands when the runnable harness changes.
- Record guessed format details as metadata fields and CLI options.
- When adding Stage 11, keep the wording focused on `1bpp mask -> 2bpp style
  layers`; do not describe it as WQY glyph-shape correction.

## Design Constraints

- Treat generated artifacts under `data/processed/` as disposable.
- Keep original font assets untouched.
- Keep heavy deep-learning frameworks out of the default install until the
  extraction path is understood.
- Use Pillow for image IO unless the project grows past its needs.
