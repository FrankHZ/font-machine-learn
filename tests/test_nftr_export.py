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

from font_machine_learn.nftr import export_font_atlas, export_target_dataset, parse_rtfn_font


class NFTRExportTest(unittest.TestCase):
    def test_parses_source_nftr_metrics_and_mapping(self) -> None:
        font = parse_rtfn_font(ROOT / "a.NFTR")
        self.assertEqual(font.cell_width, 15)
        self.assertEqual(font.cell_height, 15)
        self.assertEqual(font.bpp, 2)
        self.assertEqual(font.cell_size, 57)
        self.assertEqual(font.glyph_count, 1814)
        self.assertEqual(font.index_to_codes[5], [0x0030])
        self.assertIn(0x97B9, [code for codes in font.index_to_codes.values() for code in codes])
        self.assertEqual(font.index_to_codes[1696], [0x97B9])

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

    def test_exports_target_glyph_dataset(self) -> None:
        source = ROOT / "a.NFTR"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.cell_width, 15)
            self.assertEqual(result.cell_height, 15)
            self.assertEqual(result.bpp, 2)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertEqual(len(list((root / "target").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["glyph_count"], 1814)
            self.assertGreater(metadata["mapped_glyph_count"], 1700)
            self.assertGreater(metadata["code_count"], 1700)
            zero = metadata["glyphs"][5]
            self.assertEqual(zero["codes"], [0x0030])
            self.assertEqual(zero["chars"], ["0"])
            self.assertEqual(sum(zero["histogram"].values()), 15 * 15)


if __name__ == "__main__":
    unittest.main()
