from .base import Detector, nms

__all__ = ["Detector", "nms", "load_detector"]


def load_detector(model: str = "dfine-nano", input_size: int = 640, device: str | None = None):
    from .hf import HFDetector

    return HFDetector(model, input_size=input_size, device=device)
