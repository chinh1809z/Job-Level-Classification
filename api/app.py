from flask import Flask, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)

model = joblib.load("../models/final_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():

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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Job Career Level Classification API"
    })


if __name__ == "__main__":
    app.run(debug=True)