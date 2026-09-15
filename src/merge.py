"""All public boxes use half-open [left, top, right, bottom] pixel coordinates."""
import math
from collections.abc import Iterable

from .types import Box


def union_box(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    if not boxes:
        raise ValueError('Cannot merge an empty set of boxes')
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def clip_box(box: Iterable[float], size: tuple[int, int]) -> Box:
    x1, y1, x2, y2 = box
    width, height = size
    return (max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
            max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2))))


def normalize_box(box: Box, size: tuple[int, int]) -> list[int]:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    x1, y1, x2, y2 = clip_box(box, size)
    return [int(1000 * x1 / width), int(1000 * y1 / height),
            int(1000 * x2 / width), int(1000 * y2 / height)]


def denormalize_box(box: Box, size: tuple[int, int]) -> Box:
    width, height = size
    return clip_box((box[0] * width / 1000, box[1] * height / 1000,
                     box[2] * width / 1000, box[3] * height / 1000), size)


def vertical_overlap(a: Box, b: Box) -> float:
    return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3]-a[1], b[3]-b[1]))


def intersection_over_union(a: Box, b: Box) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / area if area else 0.0


def restore_box(box: Box, original_size: tuple[int, int], clockwise: int) -> Box:
    """Invert the exact right-angle rotation used before OCR/model inference."""
    width, height = original_size
    x1, y1, x2, y2 = box
    if clockwise == 90:
        box = (y1, height-x2, y2, height-x1)
    elif clockwise == 180:
        box = (width-x2, height-y2, width-x1, height-y1)
    elif clockwise == 270:
        box = (width-y2, x1, width-y1, x2)
    elif clockwise != 0:
        raise ValueError('Rotation must be 0, 90, 180, or 270')
    return clip_box(box, original_size)

