# Agent Notes

This repository is for experimenting with machine learning over tiny bitmap
fonts. The source target style is a 15x15, 2bpp Nintendo DS font with separate
transparent, shadow, edge, and main-stroke levels.

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
- `wqy-zenhei.ttc`: source bitmap font family; WQY Sharp face index `2` is the
  intended input font after the ML pipeline exists.
- `docs/targets.md`: target split and expected outputs.
- `docs/deliverables.md`: stage deliverables, verification, and commit rhythm.
- `scripts/export_nftr.py`: main smoke-test command.
- `scripts/extract_target_glyphs.py`: Stage 1 target glyph dataset command.
- `scripts/render_source_glyphs.py`: Stage 2 WQY Sharp source glyph command.
- `scripts/run_shadow_baseline.py`: Stage 3 rule-based transform command.
- `scripts/train_mlp_baseline.py`: Stage 4 trainable pixel-level MLP baseline.
- `scripts/tune_shadow_baseline.py`: Stage 5 visual shadow rule search.
- `scripts/check_env.py`: local dependency sanity check.
- `src/font_machine_learn/nftr.py`: parser/exporter implementation.
- `tests/test_nftr_export.py`: fixed smoke-test harness.
- `data/processed/`: default place for generated PNG/JSON outputs.

## Target Split

- Target 0: export `a.NFTR` to atlas PNG/JSON and keep the parser stable.
- Target 1: split NFTR glyphs into labeled per-glyph target records using
  `PAMC`, `HDWC`, and glyph indexes.
- Target 2: render matching source glyphs from `wqy-zenhei.ttc`, face index `2`,
  into `15x15` cells.
- Target 3: train/evaluate the first transform from WQY Sharp bitmap glyphs to
  the NFTR 2bpp shadow style.

Do not start Target 3 until Target 1 and Target 2 have repeatable metadata and
contact sheets.

## Smoke Test

Run this after changes:

```powershell
python scripts/export_nftr.py a.NFTR --out data/processed/a_atlas.png
python scripts/extract_target_glyphs.py a.NFTR
python scripts/render_source_glyphs.py
python scripts/run_shadow_baseline.py
python scripts/train_mlp_baseline.py
python scripts/tune_shadow_baseline.py
python -m unittest discover
python scripts/check_env.py
```

Expected result:

- command exits successfully;
- `data/processed/a_atlas.png` exists;
- `data/processed/a_atlas.json` exists and records the detected/export mode.
- `data/processed/glyphs/target_metadata.json` exists after Stage 1 extraction.
- `data/processed/glyphs/source_metadata.json` exists after Stage 2 rendering.
- `data/processed/glyphs/baseline_shadow_metadata.json` exists after Stage 3.
- `data/processed/glyphs/baseline_mlp_metadata.json` exists after Stage 4.
- `data/processed/glyphs/baseline_tuned_shadow_metadata.json` exists after Stage 5.
- unittest passes and confirms the source NFTR shape.

## Documentation Rules

- Update `docs/targets.md` when changing target boundaries or dataset outputs.
- Update `docs/deliverables.md` when stage deliverables or verification changes.
- Update README commands when the runnable harness changes.
- Record guessed format details as metadata fields and CLI options.

## Design Constraints

- Treat generated artifacts under `data/processed/` as disposable.
- Keep original font assets untouched.
- Keep heavy deep-learning frameworks out of the default install until the
  extraction path is understood.
- Use Pillow for image IO unless the project grows past its needs.
