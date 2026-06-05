"""SAM 2.1 segmentation from bounding-box prompts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _resolve_sam2_config(model_cfg: str) -> str:
    """Resolve a SAM 2 model config, checking local path then installed package.

    ``build_sam2`` uses Hydra internally to resolve config names from the
    installed ``sam2`` package, so this helper validates that the config
    exists *before* calling ``build_sam2`` — surfacing a clear error early.

    Args:
        model_cfg: Local file path **or** Hydra config name recognised by
            ``build_sam2`` (e.g. ``"configs/sam2.1/sam2.1_hiera_l.yaml"``).

    Returns:
        The validated config string (local path or Hydra config name).

    Raises:
        FileNotFoundError: If the config cannot be found locally or inside
            the installed ``sam2`` package.
    """
    # 1. Local file takes precedence.
    if Path(model_cfg).is_file():
        return model_cfg

    # 2. Validate the config exists inside the installed sam2 package.
    #    build_sam2 resolves configs via Hydra from the package root.
    import sam2 as _sam2_pkg

    pkg_config = Path(_sam2_pkg.__path__[0]) / model_cfg
    if pkg_config.is_file():
        return model_cfg  # Hydra resolves this internally

    raise FileNotFoundError(
        f"SAM 2 config not found at '{model_cfg}' (local) "
        f"or '{pkg_config}' (installed package). "
        "Ensure sam2 is installed or provide a valid config path/name."
    )


def load_sam2(
    weights_path: str,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
) -> SAM2ImagePredictor:
    """Load a SAM 2.1 image predictor.

    Args:
        weights_path: Path to the SAM 2.1 checkpoint file.
        model_cfg: Relative path or Hydra config name for the SAM 2.1 model
            config YAML.  Can be a local file path or a config name resolved
            from the installed ``sam2`` package (e.g.
            ``"configs/sam2.1/sam2.1_hiera_l.yaml"``).
        device: Target device (``"cuda"`` or ``"cpu"``).

    Returns:
        A :class:`SAM2ImagePredictor` ready for :meth:`set_image` / :meth:`predict`.
    """
    resolved_cfg = _resolve_sam2_config(model_cfg)
    sam_model = build_sam2(resolved_cfg, weights_path, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    return predictor


def segment_single_box(
    predictor: SAM2ImagePredictor,
    box: list[float],
) -> np.ndarray:
    """Segment a single bounding box (assumes ``set_image`` already called).

    Args:
        predictor: A SAM 2.1 predictor with the image already set.
        box: Bounding box as ``[x1, y1, x2, y2]`` in absolute pixel coords.

    Returns:
        Binary mask of shape ``(H, W)`` as a NumPy bool array — the candidate
        with the highest predicted IoU score.
    """
    box_tensor = torch.tensor([box], dtype=torch.float32, device=predictor.device)

    with torch.no_grad():
        masks, scores, _logits = predictor.predict(
            box=box_tensor,
            multimask_output=True,
        )

    # masks: (N, H, W), scores: (N,) — pick the best candidate.
    best_idx = int(scores.argmax())
    return masks[best_idx].astype(bool)


def segment_boxes(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    boxes: list[list[float]],
) -> list[np.ndarray]:
    """Produce binary masks for each bounding box in a single image.

    ``set_image`` is called **once** for the whole image; each box is then
    segmented individually via :func:`segment_single_box`.

    Args:
        predictor: A SAM 2.1 predictor (from :func:`load_sam2`).
        image: Image as a NumPy array in HWC uint8 format (RGB).
        boxes: List of bounding boxes, each ``[x1, y1, x2, y2]``.

    Returns:
        List of binary masks (one per box), each of shape ``(H, W)``
        as a NumPy bool array.  Returns an empty list when *boxes* is empty.
    """
    if not boxes:
        return []

    with torch.no_grad():
        predictor.set_image(image)

    masks: list[np.ndarray] = []
    for box in boxes:
        mask = segment_single_box(predictor, box)
        masks.append(mask)
    return masks
