import os
import pytest
from resume_screening.resume_parser import extract_pdf_text
from resume_screening.skill_extractor import extract_skills
from resume_screening.pipeline import screen_resume
from resume_screening.ranking import rank_candidates


def test_skill_extraction_multi_word():
    text = "Experienced in machine learning, deep learning, data analysis, and scikit-learn. Familiar with ci/cd and C++."
    skills = extract_skills(text)
    assert "machine learning" in skills
    assert "deep learning" in skills
    assert "data analysis" in skills
    assert "scikit-learn" in skills
    assert "ci/cd" in skills
    assert "c++" in skills


def test_pdf_ocr_extraction():
    sample_path = r"C:\Users\ChiefIT\Documents\Resume_System\personal_resume_dataset\Resumes PDF\Testing\2.pdf"
    if os.path.exists(sample_path):
        text = extract_pdf_text(sample_path)
        assert len(text) > 100
        skills = extract_skills(text)
        assert len(skills) > 0


def test_end_to_end_pipeline():
    sample_path = r"C:\Users\ChiefIT\Documents\Resume_System\personal_resume_dataset\Resumes PDF\Testing\2.pdf"
    if os.path.exists(sample_path):
        text = extract_pdf_text(sample_path)
        job_desc = "Testing Specialist with certification, communication, agile, python, and bachelor degree."
        result = screen_resume("2.pdf", text, job_desc)
        assert len(result["candidate_skills"]) > 0
        assert result["score"] > 0
        assert result["clean_text"] != ""
