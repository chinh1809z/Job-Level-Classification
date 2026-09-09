import joblib
import pandas as pd


# Load model
model = joblib.load("../models/final_model.pkl")

# New job posting
new_job = pd.DataFrame([{
    "title": "Senior Software Engineer",
    "location": "Berlin",
    "description": "We are looking for a senior software engineer to develop and maintain software applications.",
    "function": "IT",
    "industry": "Information Technology"
}])


# Prediction
prediction = model.predict(new_job)

print("Predicted career level:", prediction[0])