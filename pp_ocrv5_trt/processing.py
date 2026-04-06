"""Pre/post processing for PP-OCRv5 detection and recognition.

Matches PaddleOCR's pipeline (PaddleX DBPostProcess / CTCLabelDecode).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

# --- Detection ---

# HF Transformers image processor uses these (ImageNet-ish, BGR-order channels)
DET_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
DET_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)

# Recognition uses standard ImageNet stats
REC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
REC_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_det(
    image: np.ndarray,
    limit_side_len: int = 960,
    limit_type: str = "max",
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """Preprocess an image for detection.

    Args:
        image: BGR image (H,W,3) uint8.
        limit_side_len: Max (or min) side length.
        limit_type: "max" or "min".

    Returns:
        (tensor, original_size, scale_factor)
        tensor: (1,3,H',W') float32 normalized.
        original_size: (orig_h, orig_w).
        scale_factor: (h_scale, w_scale) from resized to original.
    """
    orig_h, orig_w = image.shape[:2]

    # Resize preserving aspect ratio, round to 32
    ratio = 1.0
    if limit_type == "max":
        if max(orig_h, orig_w) > limit_side_len:
            ratio = limit_side_len / max(orig_h, orig_w)
    else:
        if min(orig_h, orig_w) < limit_side_len:
            ratio = limit_side_len / min(orig_h, orig_w)

    new_h = int(orig_h * ratio)
    new_w = int(orig_w * ratio)
    # Round to nearest 32
    new_h = max(32, int(round(new_h / 32) * 32))
    new_w = max(32, int(round(new_w / 32) * 32))

    resized = cv2.resize(image, (new_w, new_h))

    # BGR→RGB, HWC→CHW, normalize
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - DET_MEAN) / DET_STD
    tensor = rgb.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)

    scale_factor = (orig_h / new_h, orig_w / new_w)
    return tensor, (orig_h, orig_w), scale_factor


def postprocess_det(
    seg_map: np.ndarray,
    original_size: tuple[int, int],
    scale_factor: tuple[float, float],
    threshold: float = 0.3,
    box_threshold: float = 0.6,
    unclip_ratio: float = 2.0,
    max_candidates: int = 1000,
) -> dict:
    """DBNet post-processing: seg map → bounding boxes.

    Args:
        seg_map: Model output (1,1,H,W) or (H,W) float32 probability map.
        original_size: (orig_h, orig_w).
        scale_factor: (h_scale, w_scale).
        threshold: Binarization threshold.
        box_threshold: Min box confidence.
        unclip_ratio: Polygon expansion ratio.
        max_candidates: Max number of contours to process.

    Returns:
        {"boxes": np.array(N,4,2), "scores": np.array(N,)}
        boxes are in (4-point polygon, xy coords) in original image space.
    """
    # Squeeze to 2D
    if seg_map.ndim == 4:
        seg_map = seg_map[0, 0]
    elif seg_map.ndim == 3:
        seg_map = seg_map[0]

    h, w = seg_map.shape
    h_scale, w_scale = scale_factor

    # Binarize
    binary = (seg_map > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    scores = []

    for contour in contours[:max_candidates]:
        if len(contour) < 4:
            continue

        # Score: mean of seg_map values inside the contour's bounding rect
        score = _box_score_fast(seg_map, contour)
        if score < box_threshold:
            continue

        # Unclip: expand the polygon
        poly = contour.reshape(-1, 2).astype(np.float64)
        expanded = _unclip(poly, unclip_ratio)
        if expanded is None:
            continue

        # Get minimum area rect → 4-point box
        rect = cv2.minAreaRect(expanded)
        box = cv2.boxPoints(rect)

        # Scale back to original image coordinates
        box[:, 0] = box[:, 0] * w_scale
        box[:, 1] = box[:, 1] * h_scale

        # Clip to image bounds
        box[:, 0] = np.clip(box[:, 0], 0, original_size[1])
        box[:, 1] = np.clip(box[:, 1], 0, original_size[0])

        boxes.append(box)
        scores.append(score)

    if boxes:
        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "scores": np.array(scores, dtype=np.float32),
        }
    return {"boxes": np.empty((0, 4, 2), dtype=np.float32), "scores": np.empty(0, dtype=np.float32)}


def _box_score_fast(seg_map: np.ndarray, contour: np.ndarray) -> float:
    """Compute mean score inside the contour's axis-aligned bounding rect."""
    h, w = seg_map.shape
    box = contour.reshape(-1, 2)
    xmin = np.clip(box[:, 0].min(), 0, w - 1).astype(int)
    xmax = np.clip(box[:, 0].max(), 0, w - 1).astype(int)
    ymin = np.clip(box[:, 1].min(), 0, h - 1).astype(int)
    ymax = np.clip(box[:, 1].max(), 0, h - 1).astype(int)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box - np.array([xmin, ymin])
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)

    return cv2.mean(seg_map[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _unclip(poly: np.ndarray, unclip_ratio: float) -> np.ndarray | None:
    """Expand polygon using Clipper library."""
    polygon = Polygon(poly)
    if polygon.area < 1:
        return None
    distance = polygon.area * unclip_ratio / polygon.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        [tuple(p) for p in poly.astype(int)],
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0], dtype=np.float32).reshape(-1, 1, 2)


# --- Recognition ---


def preprocess_rec(
    images: list[np.ndarray],
    target_height: int = 48,
    max_width: int = 3200,
) -> np.ndarray:
    """Preprocess cropped text images for recognition.

    Args:
        images: List of BGR crops (H,W,3) uint8.
        target_height: Resize height.
        max_width: Max width cap.

    Returns:
        Batched tensor (B,3,target_height,max_w_in_batch) float32, normalized.
    """
    processed = []
    max_w = 0

    for img in images:
        h, w = img.shape[:2]
        ratio = target_height / h
        new_w = min(int(math.ceil(w * ratio)), max_width)
        resized = cv2.resize(img, (new_w, target_height))

        # BGR→RGB, normalize with ImageNet stats
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - REC_MEAN) / REC_STD
        chw = rgb.transpose(2, 0, 1)  # (3, H, W)

        processed.append(chw)
        max_w = max(max_w, new_w)

    # Pad to max width in batch
    batch = np.zeros((len(processed), 3, target_height, max_w), dtype=np.float32)
    for i, chw in enumerate(processed):
        _, _, w = chw.shape
        batch[i, :, :, :w] = chw

    return batch


def postprocess_rec(
    logits: np.ndarray,
    character_list: list[str],
) -> list[tuple[str, float]]:
    """CTC greedy decode.

    Args:
        logits: Model output (B, T, vocab_size), post-softmax probabilities.
        character_list: Character list (index 0 = blank).

    Returns:
        List of (text, confidence) tuples.
    """
    preds_idx = logits.argmax(axis=-1)    # (B, T)
    preds_prob = logits.max(axis=-1)      # (B, T)

    results = []
    for b in range(preds_idx.shape[0]):
        indices = preds_idx[b]
        probs = preds_prob[b]

        chars = []
        char_probs = []
        prev = -1
        for t in range(len(indices)):
            idx = indices[t]
            if idx != prev and idx != 0:  # Skip duplicates and blank
                if idx < len(character_list):
                    chars.append(character_list[idx])
                    char_probs.append(probs[t])
            prev = idx

        text = "".join(chars)
        confidence = float(np.mean(char_probs)) if char_probs else 0.0
        results.append((text, confidence))

    return results


# --- Utility ---


def crop_box(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop and perspective-transform a 4-point box from an image.

    Args:
        image: Source image (H,W,3) uint8.
        box: 4 corner points (4,2) float32.

    Returns:
        Cropped and straightened image.
    """
    box = _order_points(box)
    tl, tr, br, bl = box

    width = max(
        int(np.linalg.norm(tr - tl)),
        int(np.linalg.norm(br - bl)),
    )
    height = max(
        int(np.linalg.norm(bl - tl)),
        int(np.linalg.norm(br - tr)),
    )

    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]   # top-left
    rect[2] = pts[s.argmax()]   # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[d.argmin()]   # top-right
    rect[3] = pts[d.argmax()]   # bottom-left
    return rect
