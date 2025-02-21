from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)