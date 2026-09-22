from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "of", "on", "or", "that", "the", "to", "with", "using", "this", "will", "our", "you"}

@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()
    except Exception:
        return None


def normalize_text(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[/_|]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z][a-z0-9+#.\-]*", normalized)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def preprocess_text(text: str) -> str:
    tokens = tokenize(text)
    lemma = _lemmatizer()
    if lemma:
        try:
            tokens = [lemma.lemmatize(token) for token in tokens]
        except LookupError:
            pass
    return " ".join(dict.fromkeys(tokens))
