from __future__ import annotations

import pandas as pd


def label_for_score(score: float) -> str:
    if score >= 0.75:
        return "Strong Match"
    if score >= 0.45:
        return "Moderate Match"
    return "Weak Match"


def rank_candidates(results: list[dict]) -> pd.DataFrame:
    ranked = pd.DataFrame(results).sort_values(["score", "jaccard_similarity"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
