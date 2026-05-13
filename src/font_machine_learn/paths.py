from __future__ import annotations

from pathlib import Path


GLYPH_ROOT = Path("data/processed/glyphs")

STAGE1_TARGET = GLYPH_ROOT / "stage1_target"
TARGET_DIR = STAGE1_TARGET / "target"
TARGET_METADATA = STAGE1_TARGET / "target_metadata.json"
TARGET_CONTACT = STAGE1_TARGET / "target_contact.png"

STAGE2_SOURCE = GLYPH_ROOT / "stage2_source"
SOURCE_DIR = STAGE2_SOURCE / "source"
SOURCE_METADATA = STAGE2_SOURCE / "source_metadata.json"
SOURCE_TARGET_CONTACT = STAGE2_SOURCE / "source_target_contact.png"

STAGE3_SHADOW = GLYPH_ROOT / "stage3_shadow"
BASELINE_SHADOW_DIR = STAGE3_SHADOW / "baseline_shadow"
BASELINE_SHADOW_METADATA = STAGE3_SHADOW / "baseline_shadow_metadata.json"
BASELINE_SHADOW_CONTACT = STAGE3_SHADOW / "baseline_shadow_contact.png"

STAGE4_MLP = GLYPH_ROOT / "stage4_mlp"
BASELINE_MLP_DIR = STAGE4_MLP / "baseline_mlp"
BASELINE_MLP_METADATA = STAGE4_MLP / "baseline_mlp_metadata.json"
BASELINE_MLP_CONTACT = STAGE4_MLP / "baseline_mlp_contact.png"

STAGE5_TUNED_SHADOW = GLYPH_ROOT / "stage5_tuned_shadow"
BASELINE_TUNED_SHADOW_DIR = STAGE5_TUNED_SHADOW / "baseline_tuned_shadow"
BASELINE_TUNED_SHADOW_METADATA = STAGE5_TUNED_SHADOW / "baseline_tuned_shadow_metadata.json"
BASELINE_TUNED_SHADOW_SEARCH = STAGE5_TUNED_SHADOW / "baseline_tuned_shadow_search.json"
BASELINE_TUNED_SHADOW_CONTACT = STAGE5_TUNED_SHADOW / "baseline_tuned_shadow_contact.png"

STAGE6_REVIEW = GLYPH_ROOT / "stage6_review"
REVIEW_REPORT = STAGE6_REVIEW / "review_report.json"
REVIEW_WORST_CASES = STAGE6_REVIEW / "review_worst_cases.json"
REVIEW_CONTACT = STAGE6_REVIEW / "review_contact.png"

STAGE7_BINARY = GLYPH_ROOT / "stage7_binary"
TARGET_1BPP_DIR = STAGE7_BINARY / "target_1bpp"
BINARY_DIAGNOSTIC_METADATA = STAGE7_BINARY / "binary_diagnostic_metadata.json"
BINARY_DIAGNOSTIC_CONTACT = STAGE7_BINARY / "binary_diagnostic_contact.png"

STAGE8_WEIGHT = GLYPH_ROOT / "stage8_weight"
SOURCE_WEIGHTED_DIR = STAGE8_WEIGHT / "source_weighted"
BASELINE_WEIGHTED_SHADOW_DIR = STAGE8_WEIGHT / "baseline_weighted_shadow"
WEIGHT_SEARCH_METADATA = STAGE8_WEIGHT / "weight_search_metadata.json"
WEIGHT_SEARCH_JSON = STAGE8_WEIGHT / "weight_search.json"
BASELINE_WEIGHTED_SHADOW_CONTACT = STAGE8_WEIGHT / "baseline_weighted_shadow_contact.png"

STAGE9_CJK_STYLE = GLYPH_ROOT / "stage9_cjk_style"
CJK_STYLE_DIR = STAGE9_CJK_STYLE / "baseline_cjk_style"
CJK_STYLE_METADATA = STAGE9_CJK_STYLE / "cjk_style_metadata.json"
CJK_STYLE_SEARCH = STAGE9_CJK_STYLE / "cjk_style_search.json"
CJK_STYLE_CONTACT = STAGE9_CJK_STYLE / "cjk_style_contact.png"
CJK_STYLE_WORST_CONTACT = STAGE9_CJK_STYLE / "cjk_style_worst_contact.png"

LEGACY_FLAT = GLYPH_ROOT / "legacy_flat"
