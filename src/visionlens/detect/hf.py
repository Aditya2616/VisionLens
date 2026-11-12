"""Hugging Face object detectors (D-FINE, RT-DETR) on CPU."""

from __future__ import annotations

import numpy as np

from ..schemas import Detection
from ..sysutil import resolve_device
from .base import nms

MODELS = {
    "dfine-nano": "ustc-community/dfine-nano-coco",
    "dfine-small": "ustc-community/dfine-small-coco",
    "rtdetr-r18": "PekingU/rtdetr_r18vd",
}
DEFAULT_MODEL = "dfine-nano"

# COCO classes worth tracking in a scene description. Small accessories (bags, signs, ...)
# produce a lot of flicker and are left out unless asked for.
SCENE_CLASSES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "train",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)


class HFDetector:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        input_size: int = 640,
        score_floor: float = 0.15,
        classes: frozenset[str] | None = SCENE_CLASSES,
        device: str | None = None,
    ) -> None:
        import torch  # noqa: F401  (imported here so the module imports without torch)
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.key = model
        self.model_id = MODELS.get(model, model)
        self.name = model
        self.input_size = input_size
        self.score_floor = score_floor
        self.classes = classes
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).eval()
        self.device = resolve_device(device)
        try:
            self.model.to(self.device)
        except RuntimeError:  # e.g. CUDA out of memory: stay on the CPU rather than fail
            self.device = "cpu"
            self.model.to("cpu")

    def detect(self, frame: np.ndarray | None, index: int = 0) -> list[Detection]:
        import cv2
        import torch
        from PIL import Image

        assert frame is not None
        h, w = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        s = self.input_size
        inputs = self.processor(images=image, return_tensors="pt", size={"height": s, "width": s})
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.score_floor, target_sizes=[(h, w)]
        )[0]
        id2label = self.model.config.id2label
        dets: list[Detection] = []
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"], strict=True):
            cls = id2label[int(label)]
            if self.classes is not None and cls not in self.classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(w - 1.0, x2), min(h - 1.0, y2)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            dets.append(Detection(box=(x1, y1, x2, y2), score=float(score), cls=cls))
        return nms(dets)
