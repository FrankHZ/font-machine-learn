from __future__ import annotations

from dataclasses import asdict, dataclass


LevelGrid = list[list[int]]


@dataclass(frozen=True)
class VisualMetrics:
    pixel_accuracy: float
    mean_absolute_error: float
    foreground_iou: float
    ink_f1: float
    shadow_f1: float
    weighted_similarity: float
    isolated_foreground_fraction: float
    visual_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def flatten(levels: LevelGrid) -> list[int]:
    return [value for row in levels for value in row]


def binary_f1(predicted: list[bool], target: list[bool]) -> float:
    true_positive = sum(1 for left, right in zip(predicted, target) if left and right)
    false_positive = sum(1 for left, right in zip(predicted, target) if left and not right)
    false_negative = sum(1 for left, right in zip(predicted, target) if not left and right)
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else (2 * true_positive) / denominator


def foreground_iou(predicted: list[int], target: list[int]) -> float:
    pred_fg = {i for i, value in enumerate(predicted) if value > 0}
    target_fg = {i for i, value in enumerate(target) if value > 0}
    union = pred_fg | target_fg
    return 1.0 if not union else len(pred_fg & target_fg) / len(union)


def isolated_foreground_fraction(levels: LevelGrid) -> float:
    height = len(levels)
    width = len(levels[0]) if height else 0
    foreground = 0
    isolated = 0
    for y in range(height):
        for x in range(width):
            if levels[y][x] == 0:
                continue
            foreground += 1
            has_neighbor = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    sx = x + dx
                    sy = y + dy
                    if 0 <= sx < width and 0 <= sy < height and levels[sy][sx] > 0:
                        has_neighbor = True
            if not has_neighbor:
                isolated += 1
    return 0.0 if foreground == 0 else isolated / foreground


def weighted_similarity(predicted: list[int], target: list[int]) -> float:
    total_weight = 0.0
    weighted_error = 0.0
    for left, right in zip(predicted, target):
        if right == 3:
            weight = 3.0
        elif right > 0:
            weight = 1.7
        elif left > 0:
            weight = 1.1
        else:
            weight = 0.25
        total_weight += weight
        weighted_error += weight * abs(left - right) / 3
    return 1.0 - weighted_error / total_weight


def compare_visual(predicted_levels: LevelGrid, target_levels: LevelGrid) -> VisualMetrics:
    predicted = flatten(predicted_levels)
    target = flatten(target_levels)
    if len(predicted) != len(target):
        raise ValueError("predicted and target level grids have different sizes")

    count = len(target)
    pixel_accuracy = sum(1 for left, right in zip(predicted, target) if left == right) / count
    mae = sum(abs(left - right) for left, right in zip(predicted, target)) / count
    fg_iou = foreground_iou(predicted, target)
    ink = binary_f1([value == 3 for value in predicted], [value == 3 for value in target])
    shadow = binary_f1(
        [0 < value < 3 for value in predicted],
        [0 < value < 3 for value in target],
    )
    similarity = weighted_similarity(predicted, target)
    isolated = isolated_foreground_fraction(predicted_levels)
    visual_score = (
        0.32 * ink
        + 0.26 * shadow
        + 0.18 * fg_iou
        + 0.18 * similarity
        + 0.06 * (1.0 - isolated)
    )
    return VisualMetrics(
        pixel_accuracy=pixel_accuracy,
        mean_absolute_error=mae,
        foreground_iou=fg_iou,
        ink_f1=ink,
        shadow_f1=shadow,
        weighted_similarity=similarity,
        isolated_foreground_fraction=isolated,
        visual_score=visual_score,
    )
