import joblib
import pandas as pd
from pathlib import Path


# =========================
# 1. Load model
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "final_model.pkl"
)


# =========================
# 2. Get user input
# =========================

print("=== Job Career Level Classification ===")

title = input("Job title: ")

location = input("Location: ")

description = input("Description: ")

function = input("Function: ")

industry = input("Industry: ")


# =========================
# 3. Create DataFrame
# =========================

new_job = pd.DataFrame([{
    "title": title,
    "location": location,
    "description": description,
    "function": function,
    "industry": industry
}])


# =========================
# 4. Prediction
# =========================

prediction = model.predict(new_job)


# =========================
# 5. Display result
# =========================

print("\nPredicted career level:", prediction[0])