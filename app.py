from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("crop_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        N = data["N"]
        P = data["P"]
        K = data["K"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        ph = data["ph"]
        rainfall = data["rainfall"]

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Apply scaling
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Convert NumPy scalar to native Python type for JSON serialization
        recommended_crop = prediction[0]
        if isinstance(recommended_crop, np.generic):
            recommended_crop = recommended_crop.item()

        return jsonify({"recommended_crop": recommended_crop})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
