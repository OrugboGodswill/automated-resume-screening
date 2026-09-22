from resume_screening.model import train_model

if __name__ == "__main__":
    report = train_model("data/resume_dataset.csv")
    print(f"Saved model using {report['selected_model']} with {report['samples']} samples")
