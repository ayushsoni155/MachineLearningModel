from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app)  # You can restrict origins later if needed (e.g., CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}}))

# Load the model and scaler
model = joblib.load('./Predictmodel.pkl')
scaler = joblib.load('./scaler.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        return jsonify({'placement_probability': float(probability)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# No app.run() here; Render uses gunicorn
if __name__ == '__main__':
    # For local testing only
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000 locally
    app.run(debug=False, host='0.0.0.0', port=port)
