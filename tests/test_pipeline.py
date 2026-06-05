"""Tests for vlm_pipeline.pipeline helper behavior used in result aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vlm_pipeline.pipeline import _classify_failure_type, _generate_experiment_tables


class TestFailureTaxonomy:
    def test_false_positive_priority(self) -> None:
        assert _classify_failure_type({"is_false_positive": True}) == "false_positive"

    def test_grounding_miss_when_not_detected(self) -> None:
        assert _classify_failure_type({"detected": False}) == "grounding_miss"

    def test_box_incomplete_when_box_iou_low(self) -> None:
        record = {"detected": True, "box_iou": 0.30, "mask_iou": 0.90}
        assert _classify_failure_type(record) == "box_incomplete"

    def test_mask_excessive_when_pred_area_is_larger(self) -> None:
        record = {
            "detected": True,
            "box_iou": 0.90,
            "mask_iou": 0.20,
            "pred_mask_area": 200,
            "gt_mask_area": 100,
        }
        assert _classify_failure_type(record) == "mask_excessive"

    def test_mask_incomplete_when_pred_area_is_smaller(self) -> None:
        record = {
            "detected": True,
            "box_iou": 0.90,
            "mask_iou": 0.20,
            "pred_mask_area": 80,
            "gt_mask_area": 100,
        }
        assert _classify_failure_type(record) == "mask_incomplete"


def _base_result(
    *,
    class_name: str,
    language: str,
    prompt_type: str,
    mask_iou: float,
    box_iou: float,
) -> dict:
    return {
        "image_id": 1,
        "class_name": class_name,
        "category_id": 1,
        "language": language,
        "prompt_type": prompt_type,
        "prompt_text": "dummy",
        "detected": True,
        "mask_iou": mask_iou,
        "dice": mask_iou,
        "boundary_f1": mask_iou,
        "utility_class": "good" if mask_iou >= 0.75 else "bad",
        "box_iou": box_iou,
        "is_false_positive": False,
    }


class TestGenerateExperimentTables:
    def test_en_only_run_generates_primary_e3_and_skips_language_tables(self, tmp_path: Path) -> None:
        results = [
            _base_result(class_name="cat", language="en", prompt_type="simple", mask_iou=0.80, box_iou=0.90),
            _base_result(class_name="cat", language="en", prompt_type="direct", mask_iou=0.60, box_iou=0.80),
            _base_result(class_name="dog", language="en", prompt_type="simple", mask_iou=0.70, box_iou=0.85),
            _base_result(class_name="dog", language="en", prompt_type="direct", mask_iou=0.50, box_iou=0.78),
        ]

        _generate_experiment_tables(results, str(tmp_path))

        assert (tmp_path / "e3_prompt_type.csv").exists()
        assert (tmp_path / "e3_prompt_per_class.csv").exists()
        assert (tmp_path / "e3_prompt_stats.csv").exists()
        assert (tmp_path / "e3_prompt_pairwise_dunn.csv").exists()
        assert not (tmp_path / "e2_lang_comparison.csv").exists()
        assert not (tmp_path / "e2_lang_delta.csv").exists()
        assert not (tmp_path / "e3_prompt_lang_appendix.csv").exists()

        e3 = pd.read_csv(tmp_path / "e3_prompt_type.csv")
        assert "conditional_miou_box75" in e3.columns
        assert set(e3["prompt_type"]) == {"simple", "direct"}

        stats_df = pd.read_csv(tmp_path / "e3_prompt_stats.csv")
        assert stats_df.loc[0, "test"] == "kruskal_wallis"
        assert "epsilon_squared" in stats_df.columns

        pairwise_df = pd.read_csv(tmp_path / "e3_prompt_pairwise_dunn.csv")
        assert set(["prompt_a", "prompt_b", "p_holm", "significant_005"]).issubset(pairwise_df.columns)

    def test_pt_presence_generates_e2_delta_and_appendix_tables(self, tmp_path: Path) -> None:
        results = [
            _base_result(class_name="cat", language="en", prompt_type="simple", mask_iou=0.80, box_iou=0.90),
            _base_result(class_name="cat", language="pt", prompt_type="simple", mask_iou=0.50, box_iou=0.88),
            _base_result(class_name="dog", language="en", prompt_type="simple", mask_iou=0.70, box_iou=0.85),
            _base_result(class_name="dog", language="pt", prompt_type="simple", mask_iou=0.40, box_iou=0.82),
            _base_result(class_name="cat", language="en", prompt_type="direct", mask_iou=0.60, box_iou=0.80),
            _base_result(class_name="cat", language="pt", prompt_type="direct", mask_iou=0.30, box_iou=0.78),
        ]

        _generate_experiment_tables(results, str(tmp_path))

        assert (tmp_path / "e2_lang_comparison.csv").exists()
        assert (tmp_path / "e2_lang_delta.csv").exists()
        assert (tmp_path / "e3_prompt_lang_appendix.csv").exists()

        e2_delta = pd.read_csv(tmp_path / "e2_lang_delta.csv")
        expected_columns = {
            "class_name",
            "en_miou",
            "pt_miou",
            "delta_en_minus_pt",
            "ci_lower",
            "ci_upper",
        }
        assert expected_columns.issubset(set(e2_delta.columns))
