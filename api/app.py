from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path


app = Flask(__name__)

# =========================
# Load model
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "final_model.pkl"
)


# =========================
# Web UI
# =========================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def web_predict():

    new_job = pd.DataFrame([{
        "title": request.form["title"],
        "location": request.form["location"],
        "description": request.form["description"],
        "function": request.form["function"],
        "industry": request.form["industry"]
    }])

    prediction = model.predict(new_job)

    return render_template(
        "index.html",
        prediction=prediction[0]
    )


# =========================
# REST API
# =========================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    new_job = pd.DataFrame([{
        "title": data["title"],
        "location": data["location"],
        "description": data["description"],
        "function": data["function"],
        "industry": data["industry"]
    }])

    prediction = model.predict(new_job)

    return jsonify({
        "predicted_career_level": prediction[0]
    })


# =========================
# Run server
# =========================

if __name__ == "__main__":
    app.run(debug=True)