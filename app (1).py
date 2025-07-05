from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("Flipkart_model.pkl")

@app.route("/")
def index():
    return "Flipkart Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    price = data.get("price", 0)

    # Prediction
    prediction = model.predict(np.array([[price]]))

    return jsonify({
        "predicted_rating": round(float(prediction[0]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)