import pandas as pd
import re
import joblib

from pathlib import Path

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectPercentile, chi2


BASE_DIR = Path(__file__).resolve().parent.parent


def filler_location(location):
    result = re.findall(r"\,\s[A-Z]{2}$", location)

    if len(result) > 0:
        return result[0][2:]
    else:
        return location


# =========================
# 1. Load data
# =========================

data = pd.read_excel(
    BASE_DIR / "data" / "job_classification.ods",
    engine="odf",
    dtype=str
)

data = data.dropna(axis=0)


# =========================
# 2. Split X and y
# =========================

data["location"] = data["location"].apply(filler_location)

target = "career_level"

X = data.drop(target, axis=1)
y = data[target]


# =========================
# 3. Train / Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. Preprocessing
# =========================

transformers = ColumnTransformer(
    transformers=[
        (
            "title",
            TfidfVectorizer(stop_words="english"),
            "title"
        ),
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["location"]
        ),
        (
            "description",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=0.01,
                max_df=0.95
            ),
            "description"
        ),
        (
            "function",
            OneHotEncoder(handle_unknown="ignore"),
            ["function"]
        ),
        (
            "industry",
            TfidfVectorizer(stop_words="english"),
            "industry"
        )
    ]
)


# =========================
# 5. Final model
# =========================

final_model = Pipeline(
    steps=[
        (
            "transformers",
            transformers
        ),
        (
            "feature_selector",
            SelectPercentile(
                chi2,
                percentile=5
            )
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced"
            )
        )
    ]
)


# =========================
# 6. Train
# =========================

final_model.fit(X_train, y_train)


# =========================
# 7. Evaluate
# =========================

y_predict = final_model.predict(X_test)

print(classification_report(y_test, y_predict))


# =========================
# 8. Save model
# =========================

joblib.dump(
    final_model,
    BASE_DIR / "models" / "final_model.pkl"
)

print("Model saved successfully.")