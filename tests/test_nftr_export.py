from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.nftr import export_font_atlas


class NFTRExportTest(unittest.TestCase):
    def test_exports_source_nftr_atlas_and_metadata(self) -> None:
        source = ROOT / "a.NFTR"
        self.assertTrue(source.exists(), "a.NFTR must be present for the smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            out_png = Path(tmp) / "a_atlas.png"
            config = export_font_atlas(source, out_png)
            out_json = out_png.with_suffix(".json")

            self.assertTrue(out_png.exists())
            self.assertTrue(out_json.exists())
            self.assertEqual(config.mode, "rtfn")
            self.assertEqual(config.cell_width, 15)
            self.assertEqual(config.cell_height, 15)
            self.assertEqual(config.bpp, 2)
            self.assertEqual(config.bytes_per_glyph, 57)
            self.assertEqual(config.glyph_count, 1814)

            metadata = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["markers_found"], ["RTFN", "FNIF", "PLGC", "HDWC", "PAMC"])

            with Image.open(out_png) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)


if __name__ == "__main__":
    unittest.main()
