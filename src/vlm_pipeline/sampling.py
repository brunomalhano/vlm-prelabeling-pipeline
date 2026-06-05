"""Stratified sampling of COCO val2017 images ensuring minimum coverage per target class."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO

TARGET_CLASSES = [1, 18, 17, 3, 2, 62, 44, 47, 53, 59]
# person=1, dog=18, cat=17, car=3, bicycle=2, chair=62, bottle=44, cup=47, apple=53, pizza=59

DEFAULT_ANNOTATIONS = "data/coco/annotations/instances_val2017.json"
DEFAULT_SAMPLE_OUTPUT = "data/sample/sample_500.json"


def stratified_sample(
    annotations_path: str,
    target_classes: list[int] = TARGET_CLASSES,
    min_instances: int = 30,
    sample_size: int = 500,
    seed: int = 42,
) -> list[int]:
    """Select a stratified sample of images from COCO ensuring class coverage.

    1. Load COCO annotations.
    2. For each target class, list all images containing at least one instance.
    3. Ensure the final set has at least *min_instances* instances per class.
    4. Fill up to *sample_size* with additional images prioritizing multi-class coverage.
    5. Return a sorted list of image IDs.
    """
    rng = random.Random(seed)
    coco = COCO(annotations_path)

    # Map each target class to the set of image IDs containing it
    class_to_images: dict[int, set[int]] = {}
    for cat_id in target_classes:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        img_ids = {ann["image_id"] for ann in coco.loadAnns(ann_ids)}
        class_to_images[cat_id] = img_ids

    selected: set[int] = set()

    # Phase 1: guarantee minimum instances per class
    for cat_id in target_classes:
        candidate_imgs = sorted(class_to_images[cat_id])
        rng.shuffle(candidate_imgs)

        for img_id in candidate_imgs:
            if _class_instance_count(coco, selected | {img_id}, cat_id) >= min_instances:
                selected.add(img_id)
                break
            selected.add(img_id)

        # Keep adding until we reach the threshold
        while _class_instance_count(coco, selected, cat_id) < min_instances:
            remaining = [i for i in candidate_imgs if i not in selected]
            if not remaining:
                break
            selected.add(remaining[0])

    # Phase 2: fill up to sample_size prioritizing multi-class coverage
    if len(selected) < sample_size:
        # Score each candidate image by how many target classes it covers
        all_candidate_imgs = set()
        for imgs in class_to_images.values():
            all_candidate_imgs |= imgs

        remaining_candidates = sorted(all_candidate_imgs - selected)
        rng.shuffle(remaining_candidates)

        # Sort by number of target classes covered (descending)
        def _multi_class_score(img_id: int) -> int:
            return sum(1 for cat_id in target_classes if img_id in class_to_images[cat_id])

        remaining_candidates.sort(key=_multi_class_score, reverse=True)

        for img_id in remaining_candidates:
            if len(selected) >= sample_size:
                break
            selected.add(img_id)

    # Phase 3: if still under sample_size, add any COCO images
    if len(selected) < sample_size:
        all_img_ids = sorted(set(coco.getImgIds()) - selected)
        rng.shuffle(all_img_ids)
        for img_id in all_img_ids:
            if len(selected) >= sample_size:
                break
            selected.add(img_id)

    return sorted(selected)


def _class_instance_count(coco: COCO, image_ids: set[int], cat_id: int) -> int:
    """Count how many annotation instances of *cat_id* appear across *image_ids*."""
    if not image_ids:
        return 0
    ann_ids = coco.getAnnIds(imgIds=list(image_ids), catIds=[cat_id])
    return len(ann_ids)


def save_sample(image_ids: list[int], output_path: str) -> None:
    """Save image IDs to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(image_ids, indent=2) + "\n", encoding="utf-8")


def print_coverage_report(
    coco: COCO,
    image_ids: list[int],
    target_classes: list[int] = TARGET_CLASSES,
) -> None:
    """Print a table showing class coverage in the sample vs. the full dataset."""
    img_set = set(image_ids)
    cats = {c["id"]: c["name"] for c in coco.loadCats(target_classes)}

    header = f"{'Class':<15} {'Full':>8} {'Sample':>8} {'%':>7}"
    print(header)
    print("-" * len(header))

    for cat_id in target_classes:
        name = cats[cat_id]
        full_count = len(coco.getAnnIds(catIds=[cat_id]))
        sample_count = len(coco.getAnnIds(imgIds=list(img_set), catIds=[cat_id]))
        pct = (sample_count / full_count * 100) if full_count else 0.0
        print(f"{name:<15} {full_count:>8} {sample_count:>8} {pct:>6.1f}%")

    print(f"\nTotal images in sample: {len(image_ids)}")


def _load_config_paths() -> tuple[str, str]:
    """Resolve default paths from experiment.yaml if available, else use constants."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"
    if config_path.exists():
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        annotations = cfg.get("data", {}).get("annotations", DEFAULT_ANNOTATIONS)
        sample_file = cfg.get("data", {}).get("sample_file", DEFAULT_SAMPLE_OUTPUT)
        return annotations, sample_file
    return DEFAULT_ANNOTATIONS, DEFAULT_SAMPLE_OUTPUT


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stratified COCO sampling")
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to instances_val2017.json (default: from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output sample JSON (default: from config)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load existing sample and print coverage report only",
    )
    args = parser.parse_args()

    ann_path, sample_path = _load_config_paths()
    annotations = args.annotations or ann_path
    output = args.output or sample_path

    if args.validate:
        sample_file = Path(output)
        if not sample_file.exists():
            raise SystemExit(f"Sample file not found: {output}")
        ids = json.loads(sample_file.read_text(encoding="utf-8"))
        coco = COCO(annotations)
        print_coverage_report(coco, ids)
    else:
        ids = stratified_sample(annotations)
        save_sample(ids, output)
        print(f"Saved {len(ids)} image IDs to {output}")
        coco = COCO(annotations)
        print_coverage_report(coco, ids)
