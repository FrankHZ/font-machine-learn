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

STAGE10_CJK_EDGES = GLYPH_ROOT / "stage10_cjk_edges"
CJK_EDGES_DIR = STAGE10_CJK_EDGES / "baseline_cjk_edges"
CJK_EDGES_METADATA = STAGE10_CJK_EDGES / "cjk_edges_metadata.json"
CJK_EDGES_SEARCH = STAGE10_CJK_EDGES / "cjk_edges_search.json"
CJK_EDGES_CONTACT = STAGE10_CJK_EDGES / "cjk_edges_contact.png"
CJK_EDGES_WORST_CONTACT = STAGE10_CJK_EDGES / "cjk_edges_worst_contact.png"

STAGE11_1BPP_STYLE = GLYPH_ROOT / "stage11_1bpp_style"
STYLE_INPUT_1BPP_DIR = STAGE11_1BPP_STYLE / "input_1bpp"
STYLE_BASELINE_2BPP_DIR = STAGE11_1BPP_STYLE / "baseline_2bpp"
STYLE_PAIRS_METADATA = STAGE11_1BPP_STYLE / "style_pairs_metadata.json"
STYLE_BASELINE_SEARCH = STAGE11_1BPP_STYLE / "style_baseline_search.json"
STYLE_BASELINE_CONTACT = STAGE11_1BPP_STYLE / "style_baseline_contact.png"
STYLE_BASELINE_WORST_CONTACT = STAGE11_1BPP_STYLE / "style_baseline_worst_contact.png"

STAGE12_TARGET_MASKS = GLYPH_ROOT / "stage12_target_masks"
TARGET_GE2_1BPP_DIR = STAGE12_TARGET_MASKS / "target_ge2_1bpp"
TARGET_EQ3_1BPP_DIR = STAGE12_TARGET_MASKS / "target_eq3_1bpp"
TARGET_MASK_COMPARE_METADATA = STAGE12_TARGET_MASKS / "target_mask_compare_metadata.json"
TARGET_MASK_COMPARE_CONTACT = STAGE12_TARGET_MASKS / "target_mask_compare_contact.png"

STAGE13_WQY_ALIGNMENT = GLYPH_ROOT / "stage13_wqy_alignment"
WQY_ALIGNMENT_DIR = STAGE13_WQY_ALIGNMENT / "aligned"
WQY_ALIGNMENT_METADATA = STAGE13_WQY_ALIGNMENT / "wqy_alignment_metadata.json"
WQY_ALIGNMENT_SEARCH = STAGE13_WQY_ALIGNMENT / "wqy_alignment_search.json"
WQY_ALIGNMENT_CONTACT = STAGE13_WQY_ALIGNMENT / "wqy_alignment_contact.png"

STAGE15_STYLE_MLP = GLYPH_ROOT / "stage15_style_mlp"
STYLE_MLP_METADATA = STAGE15_STYLE_MLP / "style_mlp_metadata.json"
STYLE_MLP_CONTACT = STAGE15_STYLE_MLP / "style_mlp_contact.png"

STAGE16_STYLE_MLP_TUNING = GLYPH_ROOT / "stage16_style_mlp_tuning"
STYLE_MLP_TUNING_METADATA = STAGE16_STYLE_MLP_TUNING / "style_mlp_tuning_metadata.json"
STYLE_MLP_TUNING_CONTACT = STAGE16_STYLE_MLP_TUNING / "style_mlp_tuning_contact.png"

STAGE17_BOUNDARY_RULES = GLYPH_ROOT / "stage17_boundary_rules"
BOUNDARY_RULE_DIR = STAGE17_BOUNDARY_RULES / "predicted_2bpp"
BOUNDARY_RULE_METADATA = STAGE17_BOUNDARY_RULES / "boundary_rule_metadata.json"
BOUNDARY_RULE_SEARCH = STAGE17_BOUNDARY_RULES / "boundary_rule_search.json"
BOUNDARY_RULE_CONTACT = STAGE17_BOUNDARY_RULES / "boundary_rule_contact.png"
BOUNDARY_RULE_WORST_CONTACT = STAGE17_BOUNDARY_RULES / "boundary_rule_worst_contact.png"

STAGE18_PATCH_READINESS = GLYPH_ROOT / "stage18_patch_readiness"
PATCH_READINESS_METADATA = STAGE18_PATCH_READINESS / "patch_readiness_metadata.json"
PATCH_READINESS_PATCHES = STAGE18_PATCH_READINESS / "patch_readiness_patches.jsonl"
PATCH_READINESS_CONTACT = STAGE18_PATCH_READINESS / "patch_readiness_contact.png"

STAGE19_PATCH_CLASSIFIER = GLYPH_ROOT / "stage19_patch_classifier"
PATCH_CLASSIFIER_METADATA = STAGE19_PATCH_CLASSIFIER / "patch_classifier_metadata.json"
PATCH_CLASSIFIER_CONTACT = STAGE19_PATCH_CLASSIFIER / "patch_classifier_contact.png"
PATCH_CLASSIFIER_ERROR_CONTACT = STAGE19_PATCH_CLASSIFIER / "patch_classifier_error_contact.png"

STAGE20_SHADOW_CLASSIFIER = GLYPH_ROOT / "stage20_shadow_classifier"
SHADOW_CLASSIFIER_METADATA = STAGE20_SHADOW_CLASSIFIER / "shadow_classifier_metadata.json"
SHADOW_CLASSIFIER_CONTACT = STAGE20_SHADOW_CLASSIFIER / "shadow_classifier_contact.png"
SHADOW_CLASSIFIER_ERROR_CONTACT = STAGE20_SHADOW_CLASSIFIER / "shadow_classifier_error_contact.png"

STAGE21_EXTERNAL_EVAL = GLYPH_ROOT / "stage21_external_eval"
EXTERNAL_EVAL_METADATA = STAGE21_EXTERNAL_EVAL / "external_eval_metadata.json"

STAGE22_SONG13_ADAPTER = GLYPH_ROOT / "stage22_song13_adapter"
SONG13_ADAPTER_METADATA = STAGE22_SONG13_ADAPTER / "song13_adapter_metadata.json"
SONG13_ADAPTER_CONTACT = STAGE22_SONG13_ADAPTER / "song13_adapter_contact.png"
SONG13_ADAPTER_ERROR_CONTACT = STAGE22_SONG13_ADAPTER / "song13_adapter_error_contact.png"

STAGE23_SONG13_CALIBRATED = GLYPH_ROOT / "stage23_song13_calibrated"
SONG13_CALIBRATED_METADATA = STAGE23_SONG13_CALIBRATED / "song13_calibrated_metadata.json"
SONG13_CALIBRATED_CONTACT = STAGE23_SONG13_CALIBRATED / "song13_calibrated_contact.png"
SONG13_CALIBRATED_ERROR_CONTACT = STAGE23_SONG13_CALIBRATED / "song13_calibrated_error_contact.png"

STAGE24_SONG13_ADD_ONLY = GLYPH_ROOT / "stage24_song13_add_only"
SONG13_ADD_ONLY_METADATA = STAGE24_SONG13_ADD_ONLY / "song13_add_only_metadata.json"
SONG13_ADD_ONLY_CONTACT = STAGE24_SONG13_ADD_ONLY / "song13_add_only_contact.png"
SONG13_ADD_ONLY_ERROR_CONTACT = STAGE24_SONG13_ADD_ONLY / "song13_add_only_error_contact.png"

STAGE25_SONG13_SOURCE_LOCKED = GLYPH_ROOT / "stage25_song13_source_locked"
SONG13_SOURCE_LOCKED_METADATA = STAGE25_SONG13_SOURCE_LOCKED / "song13_source_locked_metadata.json"
SONG13_SOURCE_LOCKED_SEARCH = STAGE25_SONG13_SOURCE_LOCKED / "song13_source_locked_search.json"
SONG13_SOURCE_LOCKED_CONTACT = STAGE25_SONG13_SOURCE_LOCKED / "song13_source_locked_contact.png"
SONG13_SOURCE_LOCKED_ERROR_CONTACT = STAGE25_SONG13_SOURCE_LOCKED / "song13_source_locked_error_contact.png"

LEGACY_FLAT = GLYPH_ROOT / "legacy_flat"
