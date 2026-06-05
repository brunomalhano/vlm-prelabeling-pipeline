"""CLI entry point for the VLM Pre-Labeling Pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(description="VLM Pre-Labeling Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # -- sample --------------------------------------------------------------
    sample_parser = subparsers.add_parser(
        "sample", help="Generate or validate stratified sample",
    )
    sample_parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    sample_parser.add_argument(
        "--validate", action="store_true",
        help="Validate an existing sample instead of generating a new one",
    )

    # -- run -----------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run", help="Run full experiment pipeline",
    )
    run_parser.add_argument(
        "--config", default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )

    # -- visualize -----------------------------------------------------------
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate visualization figures from result tables",
    )
    viz_parser.add_argument(
        "--tables-dir", default="results/tables",
        help="Directory containing CSV result tables",
    )
    viz_parser.add_argument(
        "--figures-dir", default="results/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "sample":
        _handle_sample(args)
    elif args.command == "run":
        _handle_run(args)
    elif args.command == "visualize":
        _handle_visualize(args)


def _handle_sample(args: argparse.Namespace) -> None:
    """Generate or validate a stratified sample."""
    from .pipeline import load_config
    from .sampling import (
        print_coverage_report,
        save_sample,
        stratified_sample,
    )

    from pycocotools.coco import COCO

    cfg = load_config(args.config)
    annotations_path: str = cfg["data"]["annotations"]
    sample_file: str = cfg["data"]["sample_file"]

    if args.validate:
        sample_path = Path(sample_file)
        if not sample_path.exists():
            print(f"Sample file not found: {sample_file}", file=sys.stderr)
            sys.exit(1)
        image_ids = json.loads(sample_path.read_text(encoding="utf-8"))
        coco = COCO(annotations_path)
        print_coverage_report(
            coco,
            image_ids,
            target_classes=[c["id"] for c in cfg["classes"]],
        )
    else:
        image_ids = stratified_sample(
            annotations_path,
            target_classes=[c["id"] for c in cfg["classes"]],
            min_instances=cfg["data"]["min_instances_per_class"],
            sample_size=cfg["data"]["sample_size"],
            seed=cfg["experiment"]["seed"],
        )
        save_sample(image_ids, sample_file)
        print(f"Sample saved: {len(image_ids)} images → {sample_file}")


def _handle_run(args: argparse.Namespace) -> None:
    """Run the full experiment pipeline."""
    from .pipeline import run_experiment

    run_experiment(args.config)


def _handle_visualize(args: argparse.Namespace) -> None:
    """Generate figures from existing result tables."""
    from .visualization import generate_all_figures

    generate_all_figures(args.tables_dir, args.figures_dir)
    print(f"Figures saved to {args.figures_dir}")


if __name__ == "__main__":
    main()
