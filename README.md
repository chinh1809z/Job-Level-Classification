# Job Career Level Classification

A Machine Learning project for predicting the career level of a job posting based on its title, description, location, function, and industry.

The project follows an end-to-end Machine Learning workflow, including data exploration, preprocessing, model comparison, class imbalance handling, error analysis, model serialization, and REST API deployment.

---

## Project Overview

Recruitment platforms contain a large number of job postings with different career levels. Automatically classifying job postings can help organize job listings and support recruitment-related applications.

The goal of this project is to build a multiclass classification model that predicts the career level of a job posting from its available information.

### Input Features

- `title` — Job title
- `location` — Job location
- `description` — Job description
- `function` — Job function
- `industry` — Industry

### Target

- `career_level`

The task is formulated as a **multiclass classification problem**.

---

## Dataset

The dataset contains **8,074 job postings** and 6 columns.

| Feature | Description |
|---|---|
| `title` | Job title |
| `location` | Job location |
| `description` | Job description |
| `function` | Job function |
| `industry` | Industry |
| `career_level` | Target career level |

The dataset consists of job postings from the German job market.

### Data Quality

During Exploratory Data Analysis:

- 8,074 rows were identified.
- 1 missing value was found.
- 21 duplicate records were identified.
- The target variable is highly imbalanced.

Some career-level classes contain very few samples, making them difficult to learn and evaluate reliably.

---

## Project Structure

```text
Job-Level-Classification/
│
├── api/
│   ├── app.py
│   └── test_api.py
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_imbalance_experiment.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Machine Learning Workflow

## 1. Exploratory Data Analysis

The first stage focuses on understanding the dataset and identifying potential problems before training the models.

Main tasks include:

- Inspecting dataset dimensions
- Checking missing values
- Detecting duplicate records
- Analyzing the target distribution
- Examining categorical features
- Investigating class imbalance

Notebook:

```text
notebooks/01_eda.ipynb
```

---

## 2. Baseline Model

A baseline Machine Learning pipeline was created to establish an initial performance benchmark.

### Text Features

TF-IDF Vectorization was applied to:

- `title`
- `description`

The description feature also uses unigram and bigram representations:

```python
ngram_range=(1, 2)
```

### Categorical Features

One-Hot Encoding was applied to:

- `location`
- `function`
- `industry`

### Baseline Classifier

The baseline model uses Logistic Regression.

Notebook:

```text
notebooks/02_baseline.ipynb
```

---

## 3. Model Comparison

Three different Machine Learning algorithms were compared using the same dataset split and preprocessing pipeline:

- Logistic Regression
- LinearSVC
- Random Forest

Because the dataset is highly imbalanced, **Macro F1-score** was used as the primary evaluation metric.

### Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.762 | 0.412 | 0.751 |
| **LinearSVC** | **0.759** | **0.668** | **0.749** |
| Random Forest | 0.760 | 0.417 | 0.730 |

Although Logistic Regression achieved slightly higher accuracy, LinearSVC achieved a substantially higher Macro F1-score.

Therefore, **LinearSVC was selected as the final model** because Macro F1 is more informative for this imbalanced multiclass classification problem.

Notebook:

```text
notebooks/03_model_comparison.ipynb
```

---

## 4. Handling Class Imbalance

The target variable contains significant class imbalance, with some career levels having very few samples.

Different approaches were investigated, including:

- Original training setup
- Class weighting
- Random oversampling

The experiments were used to understand how different imbalance-handling strategies affect classification performance, especially for minority classes.

The final model uses class weighting:

```python
LinearSVC(class_weight="balanced")
```

This gives greater importance to minority classes during training.

Notebook:

```text
notebooks/04_imbalance_experiment.ipynb
```

---

## 5. Error Analysis

A confusion matrix was used to investigate the model's predictions and identify the most common classification errors.

The main confusion occurred between:

- `bereichsleiter`
- `manager_team_leader`
- `senior_specialist_or_project_manager`

These career levels have overlapping terminology and responsibilities.

For example, job postings containing terms such as:

- Manager
- Director
- Lead
- Senior
- Project Manager

may belong to different career-level categories.

This suggests that some classification errors are caused not only by model limitations, but also by ambiguity between the career-level labels themselves.

Notebook:

```text
notebooks/05_error_analysis.ipynb
```

---

# Final Model

The final Machine Learning pipeline consists of:

```text
Job Posting
     │
     ├── Title
     ├── Description
     ├── Location
     ├── Function
     └── Industry
     │
     ▼
ColumnTransformer
     │
     ├── TF-IDF
     └── One-Hot Encoding
     │
     ▼
Feature Selection
     │
     ▼
LinearSVC
     │
     ▼
Predicted Career Level
```

The classifier uses:

```python
LinearSVC(class_weight="balanced")
```

The complete trained pipeline is serialized using `joblib`:

```text
models/final_model.pkl
```

Because the preprocessing steps are included in the saved pipeline, new job postings can be passed directly to the model without manually applying TF-IDF, One-Hot Encoding, or feature selection.

---

# Prediction

The trained model can be loaded using:

```python
import joblib

model = joblib.load("models/final_model.pkl")
```

A new job posting can be represented as a pandas DataFrame:

```python
import pandas as pd

new_job = pd.DataFrame([{
    "title": "Senior Software Engineer",
    "location": "Berlin",
    "description": "We are looking for a senior software engineer to develop software applications.",
    "function": "IT",
    "industry": "Information Technology"
}])

prediction = model.predict(new_job)

print("Predicted career level:", prediction[0])
```

The model returns the predicted career level.

The prediction functionality is also implemented in:

```text
src/predict.py
```

---

# REST API

The trained model is exposed through a REST API built with Flask.

The API provides a `/predict` endpoint that accepts job posting information in JSON format.

### Run the API

From the project root:

```bash
python api/app.py
```

The API will be available at:

```text
http://127.0.0.1:5000
```

### Health Check

Send a GET request to:

```text
GET /
```

Example response:

```json
{
    "message": "Job Career Level Classification API"
}
```

### Prediction Endpoint

Send a POST request to:

```text
POST /predict
```

Example request:

```json
{
    "title": "Senior Software Engineer",
    "location": "Berlin",
    "description": "We are looking for a senior software engineer to develop software applications.",
    "function": "IT",
    "industry": "Information Technology"
}
```

Example response:

```json
{
    "predicted_career_level": "senior_specialist_or_project_manager"
}
```

The API was tested successfully with an HTTP status code of `200`.

API files:

```text
api/app.py
api/test_api.py
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd Job-Level-Classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Flask
- Requests
- Jupyter Notebook

### Machine Learning Techniques

- TF-IDF
- One-Hot Encoding
- ColumnTransformer
- Feature Selection
- Linear Support Vector Machine
- Class Weighting
- Multiclass Classification
- Confusion Matrix
- F1-score

---

# Evaluation

The final model achieved:

| Metric | Score |
|---|---:|
| Accuracy | **0.759** |
| Macro F1 | **0.668** |
| Weighted F1 | **0.749** |

Macro F1 is considered the primary metric because the dataset contains highly imbalanced career-level classes.

---

# Key Takeaways

This project demonstrates an end-to-end Machine Learning workflow:

1. Data exploration
2. Data cleaning
3. Feature engineering
4. Text vectorization
5. Categorical feature encoding
6. Baseline modeling
7. Model comparison
8. Class imbalance analysis
9. Error analysis
10. Model serialization
11. REST API deployment

The project also demonstrates the importance of selecting evaluation metrics based on the characteristics of the dataset rather than relying solely on accuracy.

---

# Limitations

Several limitations remain in the current implementation:

- Some career-level classes contain very few training examples.
- Career-level boundaries can be semantically ambiguous.
- The current text preprocessing uses English stop-word removal, while the dataset is from the German job market.
- The dataset may not fully represent job postings from other countries or industries.

---

# Future Improvements

Possible improvements include:

- Using German-specific NLP preprocessing and stop-word lists
- Using German-language word embeddings
- Hyperparameter optimization
- Experimenting with transformer-based models
- Improving handling of extremely rare classes
- Building a web interface for the prediction API
- Containerizing the application with Docker
- Deploying the API to a cloud platform

---

## Author

**Pham Trung Chinh**
