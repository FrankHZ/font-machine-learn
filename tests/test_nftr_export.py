from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

from PIL import Image
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from font_machine_learn.nftr import export_font_atlas, export_target_dataset, parse_rtfn_font
from font_machine_learn.baseline import export_shadow_baseline
from font_machine_learn.boundary_rules import export_boundary_rules
from font_machine_learn.binary_diagnostic import export_binary_diagnostic
from font_machine_learn.char_class import classify_char, classify_glyph
from font_machine_learn.cjk_edges import export_cjk_edges_baseline
from font_machine_learn.cjk_style import export_cjk_style_baseline
from font_machine_learn.external_eval import export_external_eval
from font_machine_learn.patch_readiness import export_patch_readiness
from font_machine_learn.patch_classifier import export_patch_classifier
from font_machine_learn.review import ReviewBaseline, export_review_report
from font_machine_learn.shadow_search import export_tuned_shadow_baseline
from font_machine_learn.shadow_classifier import export_shadow_classifier
from font_machine_learn.song13_adapter import export_song13_adapter
from font_machine_learn.song13_add_only import export_song13_add_only
from font_machine_learn.song13_calibrated import export_song13_calibrated
from font_machine_learn.song13_layer_mlp import export_song13_layer_mlp
from font_machine_learn.song13_review import hole_mask, summarize_levels
from font_machine_learn.song13_source_locked import export_song13_source_locked
from font_machine_learn.source_font import export_source_dataset
from font_machine_learn.stage26_full_nftr import assign_codes, load_char_sequence, load_mapping_entries, width_from_levels
from font_machine_learn.stage26_nftr import pack_2bpp_values
from font_machine_learn.style_dataset import export_1bpp_style_dataset
from font_machine_learn.style_mlp import export_style_mlp
from font_machine_learn.target_mask_compare import export_target_mask_compare
from font_machine_learn.target_mask_compare import levels_to_threshold_mask
from font_machine_learn.trainable_baseline import export_mlp_baseline
from font_machine_learn.weight_search import export_weight_search
from font_machine_learn.wqy_alignment import export_wqy_alignment_diagnostic


RUN_SLOW_TESTS = os.environ.get("FML_RUN_SLOW_TESTS") == "1"
slow_test = unittest.skipUnless(RUN_SLOW_TESTS, "set FML_RUN_SLOW_TESTS=1 to run full stage export smoke tests")


@contextmanager
def workspace_tempdir():
    tmp_root = ROOT / ".tmp" / "tests"
    tmp_root.mkdir(parents=True, exist_ok=True)
    path = tmp_root / f"test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


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

        with workspace_tempdir() as tmp:
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

    def test_packs_2bpp_values_in_nftr_bit_order(self) -> None:
        payload = pack_2bpp_values([0, 1, 2, 3, 3, 2, 1, 0])
        self.assertEqual(payload, bytes([0x1B, 0xE4]))

        from font_machine_learn.nftr import decode_linear_2bpp

        self.assertEqual(decode_linear_2bpp(payload, 4, 2), [0, 1, 2, 3, 3, 2, 1, 0])

    def test_fullmap_char_list_and_punctuation_width_rules(self) -> None:
        map_file = ROOT / "ds_nftr" / "a.txt"
        if map_file.exists():
            entries = load_mapping_entries(map_file, 0xE800)
            self.assertEqual(len(entries), 3296)
            self.assertEqual(entries[0], (0x002C, ","))
            self.assertEqual(entries[-2:], [(0xEDDC, "，"), (0xEDDD, "；")])

        chars_file = ROOT / "ds_nftr" / "a-chars.txt"
        if chars_file.exists():
            chars = load_char_sequence(chars_file)
            self.assertEqual(len(chars), 3296)
            self.assertEqual(len(set(chars)), 3296)
            self.assertEqual(chars[-2:], ["，", "；"])

        entries = assign_codes(["A", "B"], 0xE80A)
        self.assertEqual(entries, [(0xE80B, "A"), (0xE80C, "B")])

        width = width_from_levels("，", [[0, 0, 0], [0, 3, 0], [0, 0, 0]], 15)
        self.assertEqual((width.left, width.glyph_width, width.advance), (0, 6, 6))

    def test_exports_target_glyph_dataset(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
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

    def test_target_mask_threshold_modes(self) -> None:
        levels = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
        ]
        self.assertEqual(
            levels_to_threshold_mask(levels, "visible"),
            [
                [False, True, True, True],
                [True, True, True, False],
            ],
        )
        self.assertEqual(
            levels_to_threshold_mask(levels, "ge2"),
            [
                [False, False, True, True],
                [True, True, False, False],
            ],
        )
        self.assertEqual(
            levels_to_threshold_mask(levels, "eq3"),
            [
                [False, False, False, True],
                [True, False, False, False],
            ],
        )

    def test_song13_review_hole_metrics(self) -> None:
        source_mask = [
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ]
        levels = [
            [3, 3, 3],
            [3, 1, 3],
            [3, 3, 3],
        ]
        self.assertEqual(
            hole_mask(source_mask),
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
        )
        summary = summarize_levels(source_mask, levels)
        self.assertEqual(summary["hole_pixels"], 1)
        self.assertEqual(summary["hole_fill_ratio"], 1.0)
        self.assertEqual(summary["source_deleted_ratio"], 0.0)

    @slow_test
    def test_exports_wqy_source_dataset(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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
            self.assertEqual(result.x_offset, 0)
            self.assertEqual(result.y_offset, 0)
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.rendered_count, 1814)
            self.assertEqual(len(list((root / "source").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["font_mode"], "L")
            self.assertEqual(metadata["x_offset"], 0)
            self.assertEqual(metadata["y_offset"], 0)
            widths = [glyph["ink_width"] for glyph in metadata["glyphs"] if glyph["ink_width"]]
            self.assertGreater(sum(1 for width in widths if 12 <= width <= 13), 1000)
            zero = metadata["glyphs"][5]
            self.assertEqual(zero["chars"], ["0"])
            self.assertGreater(zero["ink_width"], 0)

    @slow_test
    def test_exports_rule_based_shadow_baseline(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_trainable_mlp_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_tuned_shadow_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_builds_baseline_review_report_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_binary_diagnostic_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_weight_search_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_cjk_style_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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

    @slow_test
    def test_exports_cjk_edges_baseline_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
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
            result = export_cjk_edges_baseline(
                source_metadata=Path(source_data.metadata_json),
                out_dir=root / "cjk_edges",
                metadata_json=root / "cjk_edges_metadata.json",
                search_json=root / "cjk_edges_search.json",
                contact_sheet=root / "cjk_edges_contact.png",
                worst_contact_sheet=root / "cjk_edges_worst.png",
                search_limit=64,
                worst_count=24,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.worst_contact_sheet).exists())
            self.assertEqual(len(list((root / "cjk_edges").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertIn("edge_transition", metadata["style_constraints"])
            self.assertEqual(metadata["best_rule"]["name"], result.best_rule)

    @slow_test
    def test_exports_1bpp_style_dataset_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_1bpp_style_dataset(
                target_metadata=Path(target.metadata_json),
                input_dir=root / "input_1bpp",
                baseline_dir=root / "baseline_2bpp",
                metadata_json=root / "style_pairs_metadata.json",
                search_json=root / "style_search.json",
                contact_sheet=root / "style_contact.png",
                worst_contact_sheet=root / "style_worst.png",
                search_limit=64,
                worst_count=24,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.worst_contact_sheet).exists())
            self.assertEqual(len(list((root / "input_1bpp").glob("*.png"))), 1814)
            self.assertEqual(len(list((root / "baseline_2bpp").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "1bpp glyph mask -> NFTR-style 2bpp layered glyph")
            self.assertIn("input_1bpp_png", metadata["glyphs"][0])
            self.assertIn("label_2bpp_png", metadata["glyphs"][0])

    @slow_test
    def test_compares_target_masks_to_wqy_source_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_export = export_source_dataset(
                source,
                font,
                root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_contact.png",
            )
            result = export_target_mask_compare(
                source_metadata=Path(source_export.metadata_json),
                ge2_dir=root / "target_ge2",
                eq3_dir=root / "target_eq3",
                metadata_json=root / "mask_compare.json",
                contact_sheet=root / "mask_compare.png",
                contact_count=24,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertEqual(len(list((root / "target_ge2").glob("*.png"))), 1814)
            self.assertEqual(len(list((root / "target_eq3").glob("*.png"))), 1814)

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["mask_modes"]["ge2"], "target level >= 2")
            self.assertEqual(metadata["contact_sheet_order"][0], "wqy_source")

    @slow_test
    def test_exports_wqy_alignment_diagnostic_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        font = ROOT / "fonts" / "wqy-zenhei.ttc"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            source_export = export_source_dataset(
                source,
                font,
                root / "source",
                target_metadata=Path(target.metadata_json),
                metadata_json=root / "source_metadata.json",
                contact_sheet=root / "source_contact.png",
            )
            result = export_wqy_alignment_diagnostic(
                source_metadata=Path(source_export.metadata_json),
                out_dir=root / "alignment",
                metadata_json=root / "alignment.json",
                search_json=root / "alignment_search.json",
                contact_sheet=root / "alignment.png",
                search_limit=64,
                contact_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["target_modes"]["visible"], "target level > 0")
            self.assertIn("visible", metadata["best_by_mode"])

    @slow_test
    def test_exports_style_mlp_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_style_mlp(
                target_metadata=Path(target.metadata_json),
                out_dir=root / "style_mlp",
                metadata_json=root / "style_mlp.json",
                contact_sheet=root / "style_mlp.png",
                max_train_glyphs=32,
                hidden_units=16,
                max_iter=4,
                external_sources={},
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertIn(result.best_mode, {"visible", "ge2", "eq3"})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "target-derived 1bpp mask -> NFTR 2bpp style levels")
            self.assertIn("ge2", metadata["modes"])

    @slow_test
    def test_exports_boundary_rules_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_boundary_rules(
                target_metadata=Path(target.metadata_json),
                out_dir=root / "boundary",
                metadata_json=root / "boundary.json",
                search_json=root / "boundary_search.json",
                contact_sheet=root / "boundary_contact.png",
                worst_contact_sheet=root / "boundary_worst.png",
                search_limit=64,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "explain ge2 source pixels as level 2 vs level 3 boundary")

    @slow_test
    def test_exports_patch_readiness_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_patch_readiness(
                target_metadata=Path(target.metadata_json),
                mlp_pred_dir=root / "missing_mlp_predictions",
                metadata_json=root / "patch_readiness.json",
                patches_jsonl=root / "patches.jsonl",
                contact_sheet=root / "patches.png",
                examples_per_category=4,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertGreater(result.patch_record_count, 0)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.patches_jsonl).exists())
            self.assertTrue(Path(result.contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "patch-readiness analysis for ge2 1bpp source -> NFTR 2bpp level assignment",
            )
            self.assertEqual(metadata["mlp_prediction_available_glyphs"], 0)
            self.assertEqual(metadata["contact_sheet_order"][0], "source_ge2_patch")

    @slow_test
    def test_exports_patch_classifier_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_patch_classifier(
                target_metadata=Path(target.metadata_json),
                out_dir=root / "patch_classifier",
                metadata_json=root / "patch_classifier.json",
                contact_sheet=root / "patch_classifier.png",
                error_contact_sheet=root / "patch_classifier_errors.png",
                max_train_glyphs=24,
                hidden_units=12,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertGreater(result.train_pixel_count, 0)
            self.assertIn(result.best_model, {"logistic_balanced", "patch_mlp"})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "patch-only classifier for ge2 source pixels: target level 2 vs 3")
            self.assertIn("patch_mlp", metadata["models"])
            self.assertEqual(metadata["contact_sheet_order"][0], "source_ge2")

    @slow_test
    def test_exports_shadow_classifier_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            result = export_shadow_classifier(
                target_metadata=Path(target.metadata_json),
                out_dir=root / "shadow_classifier",
                metadata_json=root / "shadow_classifier.json",
                contact_sheet=root / "shadow_classifier.png",
                error_contact_sheet=root / "shadow_classifier_errors.png",
                max_train_glyphs=24,
                core_hidden_units=12,
                shadow_hidden_units=12,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertGreater(result.core_edge_train_pixel_count, 0)
            self.assertGreater(result.shadow_train_pixel_count, 0)
            self.assertIn(result.best_shadow_model, {"shadow_logistic_balanced", "shadow_patch_mlp"})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "separate shadow classifier combined with ge2 patch MLP core/edge classifier",
            )
            self.assertIn("shadow_patch_mlp", metadata["models"])
            self.assertEqual(metadata["contact_sheet_order"][0], "source_ge2")

    @slow_test
    def test_exports_external_eval_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_external_eval(
                target_metadata=Path(target.metadata_json),
                out_dir=root / "external_eval",
                metadata_json=root / "external_eval.json",
                external_sources={"target_visible": fake_source_json},
                max_train_glyphs=24,
                core_hidden_units=12,
                shadow_hidden_units=12,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertEqual(result.evaluated_sources, ["target_visible"])
            self.assertGreater(result.train_core_edge_pixel_count, 0)
            self.assertGreater(result.train_shadow_pixel_count, 0)
            self.assertTrue(Path(result.metadata_json).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "evaluate Stage20 two-head patch style model on external source masks")
            self.assertIn("target_visible", metadata["sources"])

    @slow_test
    def test_exports_song13_adapter_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_song13_adapter(
                target_metadata=Path(target.metadata_json),
                source_metadata=fake_source_json,
                out_dir=root / "song13_adapter",
                metadata_json=root / "song13_adapter.json",
                contact_sheet=root / "song13_adapter.png",
                error_contact_sheet=root / "song13_adapter_errors.png",
                max_train_glyphs=24,
                adapter_hidden_units=12,
                style_core_hidden_units=12,
                style_shadow_hidden_units=12,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertGreater(result.adapter_train_pixel_count, 0)
            self.assertIn(result.best_adapter_model, {"adapter_logistic_balanced", "adapter_patch_mlp"})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "adapt current Song13 1bpp source mask to target ge2 mask, then apply Stage20 two-head style model",
            )
            self.assertEqual(
                metadata["contact_sheet_order"],
                ["original_source", "adapted_ge2", "predicted_2bpp", "target_2bpp"],
            )

    @slow_test
    def test_exports_song13_calibrated_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_song13_calibrated(
                target_metadata=Path(target.metadata_json),
                source_metadata=fake_source_json,
                out_dir=root / "song13_calibrated",
                metadata_json=root / "song13_calibrated.json",
                contact_sheet=root / "song13_calibrated.png",
                error_contact_sheet=root / "song13_calibrated_errors.png",
                thresholds=[0.50, 0.65],
                max_train_glyphs=16,
                adapter_hidden_units=12,
                style_core_hidden_units=12,
                style_shadow_hidden_units=12,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertIn(result.best_adapter_model, {"adapter_logistic_balanced", "adapter_patch_mlp"})
            self.assertIn(result.best_threshold, {0.50, 0.65})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(metadata["task"], "calibrate Song13 source-mask adapter threshold before Stage20 style heads")
            self.assertEqual(
                metadata["contact_sheet_order"],
                ["original_source", "adapted_ge2", "predicted_2bpp", "target_2bpp"],
            )

    @slow_test
    def test_exports_song13_add_only_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_song13_add_only(
                target_metadata=Path(target.metadata_json),
                source_metadata=fake_source_json,
                out_dir=root / "song13_add_only",
                metadata_json=root / "song13_add_only.json",
                contact_sheet=root / "song13_add_only.png",
                error_contact_sheet=root / "song13_add_only_errors.png",
                thresholds=[0.65],
                max_train_glyphs=12,
                add_hidden_units=10,
                style_core_hidden_units=10,
                style_shadow_hidden_units=10,
                max_iter=4,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertEqual(result.best_source_deleted_ratio, 0.0)
            self.assertIn(result.best_model, {"add_constant", "add_logistic_balanced", "add_patch_mlp"})
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "source-preserving Song13 add-only ge2 adapter before Stage20 style heads",
            )
            self.assertEqual(
                metadata["contact_sheet_order"],
                ["original_source", "adapted_ge2", "predicted_2bpp", "target_2bpp"],
            )

    @slow_test
    def test_exports_song13_source_locked_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_song13_source_locked(
                source_metadata=fake_source_json,
                target_metadata=Path(target.metadata_json),
                out_dir=root / "song13_source_locked",
                metadata_json=root / "song13_source_locked.json",
                search_json=root / "song13_source_locked_search.json",
                contact_sheet=root / "song13_source_locked.png",
                error_contact_sheet=root / "song13_source_locked_errors.png",
                search_limit=64,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 1814)
            self.assertGreater(result.cjk_glyph_count, 1000)
            self.assertEqual(result.best_source_deleted_ratio, 0.0)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.search_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "source-locked Song13 1bpp mask -> NFTR-style 2bpp layer assignment",
            )
            self.assertEqual(
                metadata["contact_sheet_order"],
                ["original_source", "source_ge2", "predicted_2bpp", "target_2bpp"],
            )

    @slow_test
    def test_exports_song13_layer_mlp_smoke(self) -> None:
        source = ROOT / "a.NFTR"
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            target = export_target_dataset(
                source,
                root / "target",
                metadata_json=root / "target_metadata.json",
                contact_sheet=root / "target_contact.png",
            )
            target_metadata = json.loads(Path(target.metadata_json).read_text(encoding="utf-8"))
            fake_source = {
                "cell_width": target_metadata["cell_width"],
                "cell_height": target_metadata["cell_height"],
                "glyphs": [
                    {
                        **glyph,
                        "source_png": glyph["png"],
                        "target_png": glyph["png"],
                    }
                    for glyph in target_metadata["glyphs"]
                ],
            }
            fake_source_json = root / "fake_source.json"
            fake_source_json.write_text(json.dumps(fake_source, ensure_ascii=False), encoding="utf-8")
            result = export_song13_layer_mlp(
                target_metadata=Path(target.metadata_json),
                source_metadata=fake_source_json,
                out_dir=root / "song13_layer_mlp",
                metadata_json=root / "song13_layer_mlp.json",
                contact_sheet=root / "song13_layer_mlp.png",
                error_contact_sheet=root / "song13_layer_mlp_errors.png",
                core_thresholds=[0.50],
                shadow_thresholds=[0.50],
                max_train_glyphs=12,
                search_limit=64,
                eval_limit=256,
                core_hidden_units=10,
                shadow_hidden_units=10,
                max_iter=4,
                jobs=2,
                extra_eval_sources={"target_copy": fake_source_json},
                eval_source_jobs=2,
                worst_count=12,
            )
            self.assertEqual(result.glyph_count, 256)
            self.assertGreaterEqual(result.cjk_glyph_count, 0)
            self.assertGreater(result.core_edge_train_pixel_count, 0)
            self.assertGreater(result.shadow_train_pixel_count, 0)
            self.assertEqual(result.best_source_deleted_ratio, 0.0)
            self.assertTrue(Path(result.metadata_json).exists())
            self.assertTrue(Path(result.contact_sheet).exists())
            self.assertTrue(Path(result.error_contact_sheet).exists())

            metadata = json.loads(Path(result.metadata_json).read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["task"],
                "source-locked Song13 1bpp mask -> learned NFTR-style 2bpp layer assignment",
            )
            self.assertEqual(
                metadata["contact_sheet_order"],
                ["original_source", "source_ge2", "predicted_2bpp", "target_2bpp"],
            )
            self.assertIn("target_copy", metadata["eval_sources"])
            self.assertEqual(metadata["eval_limit"], 256)


if __name__ == "__main__":
    unittest.main()
