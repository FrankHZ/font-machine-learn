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

## Stage 4: Deep Learning Stack

Deliverable:

- choose and document the framework separately, likely PyTorch or ONNX-friendly
  tooling
- pin install instructions for the actual local Python/CUDA/CPU target
- add a small train/eval smoke test before committing heavy model code
