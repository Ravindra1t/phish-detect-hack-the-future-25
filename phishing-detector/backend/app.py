from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import datetime
import re # Keep re for any other basic checks if needed, or for more complex feature extraction later
from urllib.parse import urlparse # Useful for URL parsing
# import numpy as np # Not explicitly needed if TF-IDF output is directly consumable by XGBoost

# --- ML Model Imports & Loading ---
import joblib
# import pandas as pd # Only needed if you were doing batch processing like in your script. For single URL, not essential.

app = Flask(__name__)
CORS(app)

# --- Load Your ML Model and Vectorizer ---
# This code runs once when the Flask app starts.
XGB_MODEL_FILENAME = 'xgb_model.pkl'
TFIDF_VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

phishing_model = None
tfidf_vectorizer = None

try:
    phishing_model = joblib.load(XGB_MODEL_FILENAME)
    print(f"✅ Successfully loaded ML model: {XGB_MODEL_FILENAME}")
except FileNotFoundError:
    print(f"🚨 WARNING: Model file '{XGB_MODEL_FILENAME}' not found. API will not use ML for prediction.")
except Exception as e:
    print(f"🚨 ERROR loading ML model '{XGB_MODEL_FILENAME}': {e}")
    phishing_model = None

try:
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_FILENAME)
    print(f"✅ Successfully loaded TF-IDF vectorizer: {TFIDF_VECTORIZER_FILENAME}")
except FileNotFoundError:
    print(f"🚨 WARNING: TF-IDF vectorizer file '{TFIDF_VECTORIZER_FILENAME}' not found. ML prediction will fail.")
except Exception as e:
    print(f"🚨 ERROR loading TF-IDF vectorizer '{TFIDF_VECTORIZER_FILENAME}': {e}")
    tfidf_vectorizer = None


# --- Feature Extraction Function using TF-IDF ---
def extract_features_for_ml(url_string):
    """
    Transforms a URL string into TF-IDF features using the loaded vectorizer.
    The output is a sparse matrix suitable for the XGBoost model.
    """
    if not tfidf_vectorizer:
        raise ValueError("TF-IDF vectorizer is not loaded. Cannot extract features.")
    
    # The vectorizer expects an iterable of strings.
    # For a single URL, pass it as a list containing one string.
    url_features = tfidf_vectorizer.transform([url_string])
    return url_features


@app.route('/')
def home():
    return """
    <h1>Phishing Detection API (ML Powered - XGBoost + TF-IDF)</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/api/test">/api/test</a> - Test endpoint</li>
        <li><a href="/api/stats">/api/stats</a> - Get scan statistics</li>
        <li>POST /api/scan-url - Scan a URL (use Postman/curl)</li>
    </ul>
    """

def init_db():
    conn = sqlite3.connect('phishing_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            is_phishing INTEGER NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/api/scan-url', methods=['POST'])
def scan_url():
    try:
        data = request.get_json()
        url_to_scan = data.get('url')

        if not url_to_scan:
            return jsonify({'error': 'URL is required'}), 400

        is_phishing_result = 0
        confidence_score = 0.0
        message = "URL appears safe (default or ML error)"

        if phishing_model and tfidf_vectorizer:
            try:
                # 1. Extract features using TF-IDF
                features = extract_features_for_ml(url_to_scan)
                # print(f"TF-IDF features for {url_to_scan} (shape: {features.shape})") # For debugging

                # 2. Make prediction using XGBoost model
                # predict_proba returns probabilities for each class: [[prob_legitimate, prob_phishing]]
                prediction_proba = phishing_model.predict_proba(features)
                
                # Assuming your 'phishing' class is the second one (index 1)
                confidence_score = float(prediction_proba[0][1])

                # Define a threshold for classifying as phishing
                PHISHING_THRESHOLD = 0.5 # Adjust this threshold based on your model's performance and desired precision/recall
                is_phishing_result = 1 if confidence_score >= PHISHING_THRESHOLD else 0
                
                message = f"Detected by ML model (XGBoost + TF-IDF)"
                print(f"ML Prediction for {url_to_scan}: Phishing={bool(is_phishing_result)}, Confidence={confidence_score:.4f}")

            except Exception as e:
                print(f"🚨 Error during ML prediction for {url_to_scan}: {e}")
                message = f"Error in ML prediction: {e}. Using default."
                # Fallback values (or could use a very simple rule if needed)
                is_phishing_result = 0
                confidence_score = 0.0
        else:
            # Fallback if ML model or vectorizer is not loaded
            missing_components = []
            if not phishing_model: missing_components.append("model")
            if not tfidf_vectorizer: missing_components.append("vectorizer")
            
            print(f"ML component(s) ({', '.join(missing_components)}) not available. Using basic rule-based check for {url_to_scan}.")
            # Replace with your original simple_phishing_check or a simplified version if you want a fallback
            # is_phishing_result, confidence_score = simple_phishing_check(url_to_scan) # If you keep this function
            # For now, just a placeholder:
            is_phishing_result = 1 if "login-update-secure" in url_to_scan.lower() else 0 # Extremely basic example
            confidence_score = 0.75 if is_phishing_result else 0.1
            message = f"Fallback: Basic check (ML {', '.join(missing_components)} unavailable)"


        # Store in database
        conn = sqlite3.connect('phishing_stats.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO scans (url, is_phishing, confidence) VALUES (?, ?, ?)',
            (url_to_scan, is_phishing_result, confidence_score)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'url': url_to_scan,
            'is_phishing': bool(is_phishing_result),
            'confidence': confidence_score,
            'risk_level': 'HIGH' if confidence_score > 0.7 else ('MEDIUM' if confidence_score > 0.4 else 'LOW'),
            'message': message
        })

    except Exception as e:
        print(f"🚨 Error in /api/scan-url: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect('phishing_stats.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM scans')
        total_scans = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM scans WHERE is_phishing = 1')
        phishing_detected = cursor.fetchone()[0]
        conn.close()
        return jsonify({
            'total_scans': total_scans,
            'phishing_detected': phishing_detected,
            'safe_scans': total_scans - phishing_detected
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    model_status = "Loaded" if phishing_model else "Not Loaded"
    vectorizer_status = "Loaded" if tfidf_vectorizer else "Not Loaded"
    return jsonify({
        'message': 'Backend is working!',
        'status': 'success',
        'ml_model_status': model_status,
        'tfidf_vectorizer_status': vectorizer_status
    })

if __name__ == '__main__':
    init_db()
    print("🚀 Flask backend starting...")
    print("📡 API available at: http://localhost:5000")
    app.run(debug=True, port=5000)