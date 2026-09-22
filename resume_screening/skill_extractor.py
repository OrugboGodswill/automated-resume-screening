from __future__ import annotations

from functools import lru_cache
import json
import re
from pathlib import Path

DEFAULT_DICTIONARY = Path(__file__).resolve().parent.parent / "data" / "skills_dictionary.json"


def load_skill_dictionary(path: str | Path = DEFAULT_DICTIONARY) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def flatten_skills(dictionary: dict[str, list[str]]) -> list[str]:
    return sorted({skill.lower() for skills in dictionary.values() for skill in skills}, key=len, reverse=True)


@lru_cache(maxsize=2048)
def _build_skill_pattern(skill: str) -> re.Pattern:
    tokens = [re.escape(tok) for tok in re.split(r"[\s\-_/]+", skill) if tok]
    core = r"[\s\-_/]+".join(tokens)
    if skill in {"bachelor", "master", "certification", "statistic"}:
        core += r"(?:'s|s)?"
    return re.compile(r"(?<![a-z0-9])" + core + r"(?![a-z0-9])", re.IGNORECASE)


def extract_skills(text: str, dictionary: dict[str, list[str]] | None = None) -> set[str]:
    if not text:
        return set()
    dictionary = dictionary or load_skill_dictionary()
    found = set()
    for skill in flatten_skills(dictionary):
        pattern = _build_skill_pattern(skill)
        if pattern.search(text):
            found.add(skill)
    return found

