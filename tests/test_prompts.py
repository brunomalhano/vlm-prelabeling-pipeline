"""Tests for vlm_pipeline.prompts — prompt generation, article agreement, class coverage."""

from __future__ import annotations

from vlm_pipeline.prompts import generate_all_prompts, generate_prompt, get_prompts_for_class


class TestPromptGeneration:
    def test_total_prompt_count(self) -> None:
        """generate_all_prompts() produces 80 prompts when both languages are included.

        The full matrix (10 × 4 × 2) is retained for Appendix A runs.
        The primary experiment uses EN only (10 × 4 × 1 = 40).
        """
        prompts = generate_all_prompts()
        total = sum(
            len(types) for langs in prompts.values() for types in langs.values()
        )
        assert total == 80

    def test_en_only_prompt_count(self) -> None:
        """EN-only slice produces exactly 40 prompts (10 classes × 4 types × 1 language)."""
        prompts = generate_all_prompts()
        total_en = sum(len(types) for cls in prompts.values() for types in [cls["en"]])
        assert total_en == 40

    def test_portuguese_article_feminine(self) -> None:
        """Feminine nouns use article 'a': 'segmentar a cadeira', NOT 'segmentar o cadeira'."""
        assert "a cadeira" in generate_prompt("chair", "pt", "direct")
        assert "a cadeira" in generate_prompt("chair", "pt", "contextual")

    def test_portuguese_article_masculine(self) -> None:
        """Masculine nouns use article 'o': 'segmentar o cachorro'."""
        assert "o cachorro" in generate_prompt("dog", "pt", "direct")

    def test_english_simple_prompt(self) -> None:
        """Simple EN prompt is just the class name."""
        assert generate_prompt("person", "en", "simple") == "person"

    def test_english_direct_prompt(self) -> None:
        assert generate_prompt("car", "en", "direct") == "segment the car"

    def test_all_classes_covered(self) -> None:
        """All 10 classes are present in generated prompts."""
        prompts = generate_all_prompts()
        expected_classes = {
            "person", "dog", "cat", "car", "bicycle",
            "chair", "bottle", "cup", "apple", "pizza",
        }
        assert set(prompts.keys()) == expected_classes

    def test_all_types_and_languages(self) -> None:
        """Each class has all 4 types in both languages."""
        prompts = generate_all_prompts()
        for cls in prompts:
            for lang in ["en", "pt"]:
                for ptype in ["simple", "direct", "contextual", "object"]:
                    assert ptype in prompts[cls][lang], f"Missing {lang}/{ptype} for {cls}"

    def test_get_prompts_for_class_returns_8(self) -> None:
        """get_prompts_for_class returns 8 entries (4 types × 2 languages)."""
        result = get_prompts_for_class("person")
        assert len(result) == 8

    def test_portuguese_feminine_all(self) -> None:
        """All feminine classes use correct article."""
        fem_classes = ["person", "bicycle", "chair", "bottle", "apple", "pizza"]
        for cls in fem_classes:
            prompt = generate_prompt(cls, "pt", "direct")
            assert " a " in prompt or prompt.startswith("segmentar a"), (
                f"Wrong article for {cls}: {prompt}"
            )
