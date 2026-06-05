"""Prompt generation for the VLM pre-labeling experiment (10 classes × 4 types × 1 language).

Portuguese templates are retained for Appendix A runs
(set ``languages: ["en", "pt"]`` in ``configs/experiment.yaml``).
"""

from __future__ import annotations

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "simple": {"en": "{class_en}", "pt": "{class_pt}"},
    "direct": {"en": "segment the {class_en}", "pt": "segmentar {article} {class_pt}"},
    "contextual": {"en": "the {class_en} in the image", "pt": "{article} {class_pt} na imagem"},
    "object": {"en": "object: {class_en}", "pt": "objeto: {class_pt}"},
}

CLASS_MAP: dict[str, dict[str, str]] = {
    "person": {"pt": "pessoa", "article": "a"},
    "dog": {"pt": "cachorro", "article": "o"},
    "cat": {"pt": "gato", "article": "o"},
    "car": {"pt": "carro", "article": "o"},
    "bicycle": {"pt": "bicicleta", "article": "a"},
    "chair": {"pt": "cadeira", "article": "a"},
    "bottle": {"pt": "garrafa", "article": "a"},
    "cup": {"pt": "copo", "article": "o"},
    "apple": {"pt": "maçã", "article": "a"},
    "pizza": {"pt": "pizza", "article": "a"},
}


def generate_prompt(class_en: str, language: str, prompt_type: str) -> str:
    """Generate a single prompt string for the given class, language, and type.

    Handles Portuguese articles correctly (e.g. *segmentar a cadeira*, not *segmentar o cadeira*).

    Raises:
        KeyError: If *class_en*, *language*, or *prompt_type* is unknown.
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown prompt type: {prompt_type!r}")
    if language not in PROMPT_TEMPLATES[prompt_type]:
        raise KeyError(f"Unknown language: {language!r}")
    if class_en not in CLASS_MAP:
        raise KeyError(f"Unknown class: {class_en!r}")

    template = PROMPT_TEMPLATES[prompt_type][language]
    info = CLASS_MAP[class_en]

    return template.format(
        class_en=class_en,
        class_pt=info["pt"],
        article=info["article"],
    )


def generate_all_prompts(
    classes: list[dict[str, str]] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Generate all prompt variants for all classes.

    With the default EN-only config produces 40 prompts (10 × 4 × 1).
    When ``languages: ["en", "pt"]`` is set in the config (Appendix A),
    produces 80 prompts (10 × 4 × 2).

    Args:
        classes: Optional list of dicts with keys ``en``, ``pt``, ``article``.
                 Falls back to :data:`CLASS_MAP` when *None*.

    Returns:
        Nested dict ``{class_en: {language: {prompt_type: prompt_text}}}``.
    """
    if classes is not None:
        class_map = {c["en"]: {"pt": c["pt"], "article": c["article"]} for c in classes}
    else:
        class_map = CLASS_MAP

    result: dict[str, dict[str, dict[str, str]]] = {}
    for class_en, info in class_map.items():
        result[class_en] = {}
        for lang in ("en", "pt"):
            result[class_en][lang] = {}
            for ptype, templates in PROMPT_TEMPLATES.items():
                result[class_en][lang][ptype] = templates[lang].format(
                    class_en=class_en,
                    class_pt=info["pt"],
                    article=info["article"],
                )
    return result


def get_prompts_for_class(class_en: str) -> list[dict[str, str]]:
    """Return all prompt variants for a given class.

    Returns:
        List of dicts with keys ``language``, ``prompt_type``, ``prompt_text``.

    Raises:
        KeyError: If *class_en* is unknown.
    """
    if class_en not in CLASS_MAP:
        raise KeyError(f"Unknown class: {class_en!r}")

    prompts: list[dict[str, str]] = []
    for lang in ("en", "pt"):
        for ptype in PROMPT_TEMPLATES:
            prompts.append(
                {
                    "language": lang,
                    "prompt_type": ptype,
                    "prompt_text": generate_prompt(class_en, lang, ptype),
                }
            )
    return prompts


if __name__ == "__main__":
    all_prompts = generate_all_prompts()
    total = sum(
        len(types) for langs in all_prompts.values() for types in langs.values()
    )
    print(f"Generated {total} prompts ({len(all_prompts)} classes × "
          f"{len(PROMPT_TEMPLATES)} types × {sum(len(v) for v in next(iter(all_prompts.values())).values())} languages)\n")

    for class_en, langs in all_prompts.items():
        for lang, types in langs.items():
            for ptype, text in types.items():
                print(f"  [{lang:>2}] {ptype:<12} {class_en:<10} → {text}")
    print()

    # Spot-check article agreement
    checks = [
        ("chair", "pt", "direct", "segmentar a cadeira"),
        ("chair", "pt", "contextual", "a cadeira na imagem"),
        ("dog", "pt", "direct", "segmentar o cachorro"),
    ]
    print("Article agreement checks:")
    for cls, lang, ptype, expected in checks:
        actual = generate_prompt(cls, lang, ptype)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {cls}/{lang}/{ptype}: {actual!r} (expected {expected!r})")
