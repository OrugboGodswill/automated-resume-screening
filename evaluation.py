from resume_screening.model import train_model

if __name__ == "__main__":
    import json
    report = train_model("data/training_data.csv")
    print(json.dumps(report, indent=2))
