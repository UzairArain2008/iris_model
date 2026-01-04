from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("random_forest_iris_model.pkl")

# Iris target names
target_names = ["setosa", "versicolor", "virginica"]

@app.route("/")
def home():
    return "Iris Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting 4 features
    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])

    return jsonify({
        "prediction": target_names[prediction],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
