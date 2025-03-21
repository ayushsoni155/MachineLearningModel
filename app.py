from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
import threading
import time
import logging

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
model = joblib.load('./Predictmodel.pkl')
scaler = joblib.load('./scaler.pkl')

# Get the server URL from environment variables or use a default for local testing
SERVER_URL = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
PING_INTERVAL = 10 * 60  # 10 minutes in seconds

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

# Define a simple ping endpoint
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is awake!'}), 200

# Function to ping the server and keep it awake
def keep_server_awake():
    while True:
        try:
            response = requests.get(f"{SERVER_URL}/ping")
            logger.info(f"Pinged server at {time.ctime()}: Status {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Error pinging server at {time.ctime()}: {str(e)}")
        time.sleep(PING_INTERVAL)  # Wait 10 minutes before next ping

# Start the keep-awake thread
def start_keep_awake():
    thread = threading.Thread(target=keep_server_awake, daemon=True)
    thread.start()
    logger.info("Started keep-awake thread")

# Start the server and keep-awake mechanism
if __name__ == '__main__':
    # Start the keep-awake thread when the app starts
    start_keep_awake()
    
    # For Render deployment (uses gunicorn) and local testing
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000
    app.run(debug=False, host='0.0.0.0', port=port)
