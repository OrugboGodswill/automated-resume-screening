# Talent Lens — Resume Screening ML System

A complete Streamlit decision-support system for competency-focused resume screening. It implements PDF extraction, normalization, tokenization, stop-word removal, lemmatization when NLTK data is available, synonym-tolerant skill extraction, Jaccard and cosine similarity, feature engineering, supervised model selection, persisted inference, ranking, explainability, evaluation, and CSV export.

## Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The included `data/resume_dataset.csv` is the Resume Dataset format with `Category` and `Resume` columns. The trainer now adapts this schema automatically: `Category` becomes the role context, `Resume` becomes the candidate text, and transparent competency-coverage terciles create weak, moderate, and strong training labels. The original `data/training_data.csv` labeled format remains supported.

## Data contract

The preferred dataset contains `Category,Resume`. Alternatively, labeled training data may contain `job_title`, `job_description`, `resume_text`, and integer `label` columns, where `0 = weak`, `1 = moderate`, and `2 = strong`. Resume CSVs should be UTF-8 encoded; the loader handles missing rows and normalizes column names.

## Responsible use

Do not use protected characteristics or proxies for them. Resumes contain personally identifiable information, so access, retention, and exports should be governed by the organization. Results are recommendations for trained HR reviewers, never automatic hiring decisions.
