from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FEATURE_NAMES, build_features, vectorize
from .skill_extractor import extract_skills

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resume_screening_model.joblib"


def normalize_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Accept the original labeled schema or the uploaded Category/Resume corpus."""
    columns = {column.strip().lower(): column for column in data.columns}
    if {"job_description", "resume_text", "label"}.issubset(columns):
        return data.rename(columns={columns["job_description"]: "job_description", columns["resume_text"]: "resume_text", columns["label"]: "label"})
    if {"category", "resume"}.issubset(columns):
        normalized = data.rename(columns={columns["category"]: "job_title", columns["resume"]: "resume_text"})
        normalized = normalized.dropna(subset=["job_title", "resume_text"]).copy()
        profiles = normalized.groupby("job_title")["resume_text"].transform(lambda values: " ".join(values.astype(str)))
        normalized["job_description"] = profiles
        ratios = []
        for row in normalized.itertuples(index=False):
            required = extract_skills(row.job_description)
            candidate = extract_skills(row.resume_text)
            ratios.append(len(required & candidate) / len(required) if required else len(row.resume_text.split()) / 1000)
        labels = pd.qcut(pd.Series(ratios), q=3, labels=False, duplicates="drop")
        if labels.nunique(dropna=True) < 2:
            lengths = normalized["resume_text"].astype(str).str.len()
            labels = pd.qcut(lengths.rank(method="first"), q=3, labels=False, duplicates="drop")
        normalized["label"] = labels.fillna(0).astype(int)
        return normalized
    raise ValueError("Training data must contain job_description/resume_text/label or Category/Resume columns")


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for row in data.itertuples(index=False):
        job_skills = extract_skills(row.job_description)
        resume_skills = extract_skills(row.resume_text)
        rows.append(build_features(row.job_description, row.resume_text, job_skills, resume_skills))
    return pd.DataFrame(rows, columns=FEATURE_NAMES), data["label"].astype(int)


def train_model(csv_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    data = normalize_training_data(pd.read_csv(csv_path))
    data = data.dropna(subset=["job_description", "resume_text", "label"])
    features, labels = prepare_training_data(data)
    stratify = labels if labels.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=stratify)
    candidates = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
    }
    best_name, best_model, best_f1 = None, None, -1.0
    reports = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        score = f1_score(y_test, predictions, average="weighted", zero_division=0)
        reports[name] = {"accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "f1": score}
        if score > best_f1:
            best_name, best_model, best_f1 = name, model, score
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_names": FEATURE_NAMES, "model_name": best_name, "reports": reports}, model_path)
    return {"selected_model": best_name, "reports": reports, "samples": len(data)}


def ensure_model(model_path: str | Path = MODEL_PATH) -> dict:
    if not Path(model_path).exists():
        return train_model(Path(__file__).resolve().parent.parent / "data" / "resume_dataset.csv", model_path)
    return joblib.load(model_path)
