from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_NAMES = ["required_skills", "matched_skills", "missing_skills", "skill_match_ratio", "jaccard_similarity", "cosine_similarity", "years_experience", "degree_match", "certification_count", "text_length"]


def _years(text: str) -> float:
    values = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text.lower())]
    return max(values, default=0.0)


def build_features(job_text: str, resume_text: str, job_skills: Iterable[str], resume_skills: Iterable[str]) -> dict[str, float]:
    required, candidate = set(job_skills), set(resume_skills)
    union = required | candidate
    joined = [job_text, resume_text]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(joined)
        cosine = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except ValueError:
        cosine = 0.0
    return {
        "required_skills": float(len(required)),
        "matched_skills": float(len(required & candidate)),
        "missing_skills": float(len(required - candidate)),
        "skill_match_ratio": float(len(required & candidate) / len(required)) if required else 0.0,
        "jaccard_similarity": float(len(required & candidate) / len(union)) if union else 0.0,
        "cosine_similarity": cosine,
        "years_experience": _years(resume_text),
        "degree_match": float(any(term in resume_text.lower() for term in ("bachelor", "master", "phd", "doctorate", "degree", "b.sc", "bsc", "m.sc", "msc", "b.tech", "m.tech", "b.eng", "m.eng", "bba", "mba", "diploma", "associate"))),
        "certification_count": float(len(re.findall(r"\b(certified|certification|license|licensed|licensure|pmp|cpa|acca|cfa|cissp|ccna|ccnp|comptia|aws certified|azure certified|ase|osha|pe|journeyman|six sigma)\b", resume_text.lower()))),
        "text_length": float(len(resume_text.split())),
    }


import pandas as pd


def vectorize(features: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

