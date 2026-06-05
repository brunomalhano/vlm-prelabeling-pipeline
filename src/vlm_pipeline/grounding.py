"""Grounding DINO inference for text-conditioned object detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from groundingdino.util.inference import load_model, predict

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25


def _resolve_gdino_config(config_path: str) -> str:
    """Resolve a GroundingDINO config to an existing filesystem path.

    Checks for a local file first; falls back to the config bundled with
    the installed ``groundingdino`` package.

    Args:
        config_path: Local path **or** config filename (e.g.
            ``"GroundingDINO_SwinB_cfg.py"``).

    Returns:
        Absolute filesystem path to the resolved config file.

    Raises:
        FileNotFoundError: If the config cannot be found locally or in the
            installed package.
    """
    if Path(config_path).is_file():
        return config_path

    import groundingdino

    pkg_config = Path(groundingdino.__file__).parent / "config" / Path(config_path).name
    if pkg_config.is_file():
        return str(pkg_config)

    raise FileNotFoundError(
        f"GroundingDINO config not found at '{config_path}' (local) "
        f"or '{pkg_config}' (installed package). "
        "Ensure groundingdino-py is installed or provide a valid config path."
    )


def load_grounding_dino(
    weights_path: str,
    config_path: str,
    device: str = "cuda",
) -> object:
    """Load a Grounding DINO Swin-B model.

    Args:
        weights_path: Path to the ``.pth`` checkpoint file.
        config_path: Path to the GroundingDINO config (``.py``).  Can be a
            local file path or a config filename that will be resolved from
            the installed ``groundingdino`` package.
        device: Target device (``"cuda"`` or ``"cpu"``).

    Returns:
        The loaded GroundingDINO model ready for inference.
    """
    resolved_config = _resolve_gdino_config(config_path)
    model = load_model(resolved_config, weights_path, device=device)
    return model


def _cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized cxcywh boxes to absolute xyxy pixel coordinates.

    Args:
        boxes: Tensor of shape ``(N, 4)`` in center-x/y/w/h normalised format.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tensor of shape ``(N, 4)`` with absolute ``[x1, y1, x2, y2]`` values.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return torch.stack([x1, y1, x2, y2], dim=-1)


def detect(
    model: object,
    image: np.ndarray,
    prompt: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> list[dict]:
    """Run Grounding DINO detection on a single image with one text prompt.

    Args:
        model: A loaded GroundingDINO model (from :func:`load_grounding_dino`).
        image: Image as a NumPy array in HWC uint8 format (BGR or RGB).
        prompt: Text prompt describing the objects to detect.
        box_threshold: Minimum box confidence to keep a detection.
        text_threshold: Minimum text-similarity score to keep a detection.

    Returns:
        List of detections, each a dict with keys ``"box"`` (``[x1, y1, x2, y2]``
        in absolute pixel coordinates) and ``"confidence"`` (float).
    """
    import groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    h, w = image.shape[:2]

    # GroundingDINO expects a PIL-based transform pipeline.
    pil_image = PILImage.fromarray(image)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transformed_image, _ = transform(pil_image, None)

    with torch.no_grad():
        boxes, logits, _phrases = predict(
            model=model,
            image=transformed_image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    # boxes from predict() are normalised cxcywh — convert to absolute xyxy.
    abs_boxes = _cxcywh_to_xyxy(boxes, w, h)

    detections: list[dict] = []
    for box, score in zip(abs_boxes, logits):
        detections.append({
            "box": box.tolist(),
            "confidence": round(float(score), 4),
        })
    return detections


def detect_batch(
    model: object,
    image: np.ndarray,
    prompts: list[str],
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> dict[str, list[dict]]:
    """Run detection for multiple text prompts on the same image.

    The image preprocessing is done once and the model is invoked once per
    prompt, avoiding redundant image loading.

    Args:
        model: A loaded GroundingDINO model.
        image: Image as a NumPy array in HWC uint8 format.
        prompts: List of text prompts to evaluate.
        box_threshold: Minimum box confidence to keep a detection.
        text_threshold: Minimum text-similarity score to keep a detection.

    Returns:
        Dict mapping each prompt string to its list of detections.
    """
    results: dict[str, list[dict]] = {}
    for prompt in prompts:
        results[prompt] = detect(
            model, image, prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    return results
