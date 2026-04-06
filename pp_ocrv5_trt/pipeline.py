"""End-to-end OCR pipeline: image → text with TRT acceleration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .processing import (
    crop_box,
    postprocess_det,
    postprocess_rec,
    preprocess_det,
    preprocess_rec,
)
from .runtime import TrtModel

logger = logging.getLogger(__name__)


class OCRPipeline:
    """End-to-end OCR: detection → crop → recognition → text.

    Usage::

        pipeline = OCRPipeline(
            det_engine="engines/server_det.trt",
            rec_engine="engines/server_rec.trt",
            character_list_path="chars.txt",
        )
        results = pipeline(image)
        for r in results:
            print(r["text"], r["confidence"], r["box"])
    """

    def __init__(
        self,
        det_engine: str | Path,
        rec_engine: str | Path,
        character_list_path: str | Path,
        det_limit_side_len: int = 960,
        det_threshold: float = 0.3,
        det_box_threshold: float = 0.6,
        det_unclip_ratio: float = 2.0,
        rec_target_height: int = 48,
        rec_batch_size: int = 32,
    ):
        self.det = TrtModel(det_engine)
        self.rec = TrtModel(rec_engine)
        self.character_list = self._load_character_list(character_list_path)

        self.det_limit_side_len = det_limit_side_len
        self.det_threshold = det_threshold
        self.det_box_threshold = det_box_threshold
        self.det_unclip_ratio = det_unclip_ratio
        self.rec_target_height = rec_target_height
        self.rec_batch_size = rec_batch_size

    def __call__(self, image: np.ndarray) -> list[dict]:
        """Run OCR on a single image.

        Args:
            image: BGR image (H,W,3) uint8.

        Returns:
            List of dicts with keys: "text", "confidence", "box".
            Sorted top-to-bottom, left-to-right.
        """
        # Detection
        det_input, orig_size, scale_factor = preprocess_det(
            image, self.det_limit_side_len
        )
        det_output = self.det(det_input)
        det_result = postprocess_det(
            det_output,
            orig_size,
            scale_factor,
            self.det_threshold,
            self.det_box_threshold,
            self.det_unclip_ratio,
        )

        boxes = det_result["boxes"]
        det_scores = det_result["scores"]

        if len(boxes) == 0:
            return []

        # Crop text regions
        crops = []
        for box in boxes:
            crop = crop_box(image, box)
            if crop.size > 0:
                crops.append(crop)

        if not crops:
            return []

        # Recognition (in batches)
        all_texts = []
        all_confidences = []
        for i in range(0, len(crops), self.rec_batch_size):
            batch_crops = crops[i : i + self.rec_batch_size]
            rec_input = preprocess_rec(batch_crops, self.rec_target_height)
            rec_output = self.rec(rec_input)
            batch_results = postprocess_rec(rec_output, self.character_list)
            for text, conf in batch_results:
                all_texts.append(text)
                all_confidences.append(conf)

        # Assemble results
        results = []
        for idx in range(min(len(boxes), len(all_texts))):
            results.append({
                "text": all_texts[idx],
                "confidence": all_confidences[idx],
                "box": boxes[idx].tolist(),
                "det_score": float(det_scores[idx]),
            })

        # Sort top-to-bottom, left-to-right
        results.sort(key=lambda r: (min(p[1] for p in r["box"]), min(p[0] for p in r["box"])))

        return results

    @staticmethod
    def _load_character_list(path: str | Path) -> list[str]:
        """Load character dictionary. Index 0 is blank."""
        path = Path(path)
        chars = [""]  # blank at index 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    chars.append(line)
        logger.info("Loaded %d characters from %s", len(chars), path)
        return chars
