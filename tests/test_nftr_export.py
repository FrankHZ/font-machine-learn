from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from PIL import Image
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.nftr import export_font_atlas, export_target_dataset, parse_rtfn_font
from font_machine_learn.baseline import export_shadow_baseline
from font_machine_learn.binary_diagnostic import export_binary_diagnostic
from font_machine_learn.char_class import classify_char, classify_glyph
from font_machine_learn.cjk_style import export_cjk_style_baseline
from font_machine_learn.review import ReviewBaseline, export_review_report
from font_machine_learn.shadow_search import export_tuned_shadow_baseline
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.trainable_baseline import export_mlp_baseline
from font_machine_learn.weight_search import export_weight_search


class NFTRExportTest(unittest.TestCase):
    def test_classifies_glyph_characters(self) -> None:
        self.assertEqual(classify_char("漢"), "cjk")
        self.assertEqual(classify_char("あ"), "kana")
        self.assertEqual(classify_char("A"), "latin")
        self.assertEqual(classify_char("7"), "digit")
        self.assertEqual(classify_char("。"), "punct")
        self.assertEqual(classify_glyph(["漢"]), "cjk")
        self.assertEqual(classify_glyph([]), "unmapped")

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

    def test_exports_wqy_source_dataset(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            self.assertEqual(result.font_name, ("WenQuanYi Zen Hei Sharp", "Regular"))
            self.assertEqual(result.font_index, 2)
            self.assertEqual(result.font_size, 13)
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.rendered_count, 1814)
            self.assertEqual(len(list((root / "source").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            widths = [glyph["ink_width"] for glyph in metadata["glyphs"] if glyph["ink_width"]]
            self.assertGreater(sum(1 for width in widths if 12 <= width <= 13), 1000)
            zero = metadata["glyphs"][5]
            self.assertEqual(zero["chars"], ["0"])
            self.assertGreater(zero["ink_width"], 0)

    def test_exports_rule_based_shadow_baseline(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            baseline = export_shadow_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "baseline",
                metadata_json=root / "baseline_metadata.json",
                contact_sheet=root / "baseline_contact.png",
            )
            self.assertEqual(baseline.glyph_count, 1814)
            self.assertTrue(Path(baseline.metadata_json).exists())
            self.assertTrue(Path(baseline.contact_sheet).exists())
            self.assertEqual(len(list((root / "baseline").glob("*.png"))), 1814)
            self.assertGreater(baseline.mean_pixel_accuracy, 0.40)
            self.assertGreater(baseline.mean_foreground_iou, 0.20)

            metadata = json.loads(Path(baseline.metadata_json).read_text(encoding="utf-8"))
            zero = metadata["glyphs"][5]
            self.assertEqual(zero["chars"], ["0"])
            self.assertGreaterEqual(zero["pixel_accuracy"], 0.0)
            self.assertLessEqual(zero["pixel_accuracy"], 1.0)

    def test_exports_trainable_mlp_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = export_mlp_baseline(
                    source_metadata=Path(source_data.metadata_json),
                    out_dir=root / "mlp",
                    metadata_json=root / "mlp_metadata.json",
                    contact_sheet=root / "mlp_contact.png",
                    max_train_glyphs=32,
                    hidden_units=12,
                    max_iter=4,
                )
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.train_glyph_count, 32)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertEqual(len(list((root / "mlp").glob("*.png"))), 1814)
            self.assertGreaterEqual(result.mean_pixel_accuracy, 0.0)
            self.assertLessEqual(result.mean_pixel_accuracy, 1.0)

    def test_exports_tuned_shadow_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            result = export_tuned_shadow_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "tuned",
                metadata_json=root / "tuned_metadata.json",
                search_json=root / "tuned_search.json",
                contact_sheet=root / "tuned_contact.png",
                search_limit=64,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertEqual(len(list((root / "tuned").glob("*.png"))), 1814)
            self.assertGreaterEqual(result.mean_visual_score, 0.0)
            self.assertLessEqual(result.mean_visual_score, 1.0)

            search = json.loads(Path(result.search_json).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(search["rules"]), 4)
            self.assertEqual(search["rules"][0]["name"], result.best_rule)

    def test_builds_baseline_review_report_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            shadow = export_shadow_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "shadow",
                metadata_json=root / "shadow_metadata.json",
                contact_sheet=root / "shadow_contact.png",
            )
            tuned = export_tuned_shadow_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "tuned",
                metadata_json=root / "tuned_metadata.json",
                search_json=root / "tuned_search.json",
                contact_sheet=root / "tuned_contact.png",
                search_limit=64,
            )
            review = export_review_report(
                source_metadata=Path(source_data.metadata_json),
                baselines=[
                    ReviewBaseline("shadow", Path(shadow.metadata_json)),
                    ReviewBaseline("tuned", Path(tuned.metadata_json)),
                ],
                review_json=root / "review.json",
                worst_cases_json=root / "worst.json",
                contact_sheet=root / "review.png",
                worst_count=24,
                columns=4,
            )
            self.assertEqual(review.glyph_count, 1814)
            self.assertEqual(review.selected_count, 24)
            self.assertEqual(review.baselines, ["shadow", "tuned"])
            self.assertTrue(Path(review.review_json).exists())
            self.assertTrue(Path(review.worst_cases_json).exists())
            self.assertTrue(Path(review.contact_sheet).exists())

            report = json.loads(Path(review.review_json).read_text(encoding="utf-8"))
            self.assertEqual(report["image_order"], ["source", "shadow", "tuned", "target"])
            self.assertIn("visual_score", report["summaries"]["shadow"])

    def test_exports_binary_diagnostic_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            shadow = export_shadow_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "shadow",
                metadata_json=root / "shadow_metadata.json",
                contact_sheet=root / "shadow_contact.png",
            )
            result = export_binary_diagnostic(
                source_metadata=Path(source_data.metadata_json),
                baseline_metadata={"source": Path(source_data.metadata_json), "shadow": Path(shadow.metadata_json)},
                out_dir=root / "target_1bpp",
                metadata_json=root / "binary_metadata.json",
                contact_sheet=root / "binary_contact.png",
                worst_count=24,
                columns=4,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.baselines, ["source", "shadow"])
            self.assertEqual(len(list((root / "target_1bpp").glob("*.png"))), 1814)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["image_order"], ["source", "shadow", "target_1bpp"])
            self.assertIn("foreground_f1", metadata["summaries"]["source"])

    def test_exports_weight_search_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            result = export_weight_search(
                source_metadata=Path(source_data.metadata_json),
                source_out_dir=root / "weighted_source",
                shadow_out_dir=root / "weighted_shadow",
                metadata_json=root / "weight_metadata.json",
                search_json=root / "weight_search.json",
                contact_sheet=root / "weighted_contact.png",
                search_limit=64,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertEqual(len(list((root / "weighted_source").glob("*.png"))), 1814)
            self.assertEqual(len(list((root / "weighted_shadow").glob("*.png"))), 1814)
            self.assertGreaterEqual(result.source_foreground_f1, 0.0)
            self.assertLessEqual(result.source_foreground_f1, 1.0)

            search = json.loads(Path(result.search_json).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(search["rules"]), 9)
            self.assertEqual(search["rules"][0]["name"], result.best_rule)

    def test_exports_cjk_style_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "wqy-zenhei.ttc"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_data = export_source_dataset(
                nftr_source=source,
                font_path=font,
                out_dir=root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_target_contact.png",
            )
            result = export_cjk_style_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "cjk_style",
                metadata_json=root / "cjk_style_metadata.json",
                search_json=root / "cjk_style_search.json",
                contact_sheet=root / "cjk_style_contact.png",
                worst_contact_sheet=root / "cjk_style_worst.png",
                search_limit=256,
                worst_count=24,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.worst_contact_sheet).exists())
            self.assertEqual(len(list((root / "cjk_style").glob("*.png"))), 1814)
            self.assertGreaterEqual(result.cjk_visual_score, 0.0)
            self.assertLessEqual(result.cjk_visual_score, 1.0)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertIn("cjk", metadata["groups"])
            self.assertIn("non_cjk", metadata["groups"])
            self.assertEqual(metadata["best_rule"]["name"], result.best_rule)


if __name__ == "__main__":
    unittest.main()
