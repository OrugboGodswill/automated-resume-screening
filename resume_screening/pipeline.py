from __future__ import annotations

from .feature_engineering import build_features, vectorize
from .model import MODEL_PATH, ensure_model
from .preprocess import preprocess_text
from .ranking import label_for_score
from .skill_extractor import extract_skills, load_skill_dictionary


def screen_resume(filename: str, resume_text: str, job_text: str, model_bundle=None) -> dict:
    model_bundle = model_bundle or ensure_model(MODEL_PATH)
    clean_job, clean_resume = preprocess_text(job_text), preprocess_text(resume_text)
    dictionary = load_skill_dictionary()
    job_skills = extract_skills(job_text, dictionary) | extract_skills(clean_job, dictionary)
    resume_skills = extract_skills(resume_text, dictionary) | extract_skills(clean_resume, dictionary)
    features = build_features(job_text, resume_text, job_skills, resume_skills)
    model = model_bundle["model"]
    vectorized_features = vectorize(features)
    probabilities = model.predict_proba(vectorized_features)[0] if hasattr(model, "predict_proba") else None
    prediction = int(model.predict(vectorized_features)[0])
    if probabilities is not None:
        classes = list(model.classes_)
        if classes == [0, 1, 2]:
            score = float(0.0 * probabilities[0] + 0.5 * probabilities[1] + 1.0 * probabilities[2])
        elif 1 in classes:
            score = float(probabilities[classes.index(1)])
        else:
            score = float(max(probabilities))
    else:
        score = float(prediction) / max(1, max(getattr(model, "classes_", [1])))
    score = max(0.0, min(1.0, score))
    return {
        "candidate": filename.rsplit(".", 1)[0].replace("_", " "),
        "filename": filename,
        "score": score,
        "prediction": label_for_score(score),
        "matched_skills": sorted(job_skills & resume_skills),
        "missing_skills": sorted(job_skills - resume_skills),
        "candidate_skills": sorted(resume_skills),
        "features": features,
        "clean_text": clean_resume or resume_text,
        "years_experience": features["years_experience"],
        "degree_match": bool(features["degree_match"]),
        "jaccard_similarity": features["jaccard_similarity"],
    }

