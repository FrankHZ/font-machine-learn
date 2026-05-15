# Copyright and License Notes

This repository contains code and generated bitmap-font artifacts for the Font
Machine Learn experiment.

## WenQuanYi Fonts

The source fonts in `fonts/` are WenQuanYi font files. WenQuanYi documents GPL
licensing at:

- http://wenq.org/wqy2/index.cgi?GPL

This repository includes the GPL-2.0 license text at:

- `LICENSES/GPL-2.0.txt`

Release font packages generated from the WenQuanYi fonts are distributed under
GPL-2.0-only.

## Local Target NFTR

The local target `a.NFTR` and generated NFTR files are not distributed in this
repository. They are ignored by `.gitignore` and must be supplied locally by a
user who has the right to use them.

Generated public comparison images intentionally omit the proprietary target
NFTR glyph row.

## Release Format

The recommended public release artifact is:

```text
AngelCode BMFont .fnt + RGBA PNG atlas + JSON metadata
```

This format preserves the four rendered levels used by the project:

- transparent background
- shadow
- edge/anti-alias transition
- main stroke core

BDF/PCF-style 1bpp formats are not used for the release because they cannot
represent the shadow and intermediate edge layer.
