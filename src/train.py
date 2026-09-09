import pandas as pd
import re
import pickle

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectPercentile, chi2


def filler_location(location):
    result = re.findall("\\,\\s[A-Z]{2}$", location)
    if(len(result) > 0):
        return result[0][2:]
    else:
        return location


# =========================
# 1. Load data
# =========================
data = pd.read_excel("../data/job_classification.ods", engine = "odf", dtype = str)
data = data.dropna(axis = 0)



# =========================
# 2. Split x and y
# =========================
data["location"] = data["location"].apply(filler_location)
target = "career_level"
x = data.drop(target, axis = 1)
y = data[target]

# print(y.value_counts())
# print("------------------------------------")



# =========================
# 3. Train / Test split
# =========================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)



# =========================
# 4. Preprocessing
# =========================



transformers = ColumnTransformer(transformers = [
    ("title", TfidfVectorizer(stop_words = "english"), "title"),
    ("location", OneHotEncoder(handle_unknown = "ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words = "english", ngram_range=(1,2), min_df = 0.01, max_df = 0.95), "description"),
    ("function", OneHotEncoder(), ["function"]),
    ("industry", TfidfVectorizer(stop_words = "english"), "industry")
])



# =========================
# 5. Final model
# =========================
final_model = Pipeline(steps = [
    ("transformers", transformers),
    ("feature_selector", SelectPercentile(chi2, percentile = 5)),
    ("classifier", LinearSVC(class_weight = "balanced")),
])



# =========================
# 6. Train
# =========================
final_model.fit(x_train, y_train)


y_predict = final_model.predict(x_test)

# =========================
# 7. Evaluate
# =========================
print(classification_report(y_test, y_predict))


# =========================
# 8. Save model
# =========================
# filename = 'finalized_model.pkl'
# pickle.dump(final_model, open(filename, 'wb'))
