#!/usr/bin/env python3
"""
Diabetes Prediction Model Serving Script
Serves the trained Random Forest model for diabetes prediction
"""

import os
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiabetesPredictionHandler(BaseHTTPRequestHandler):
    """HTTP handler for diabetes prediction requests"""
    
    def __init__(self, *args, model=None, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self._handle_health_check()
        elif parsed_path.path == '/info':
            self._handle_info()
        elif parsed_path.path == '/predict' and parsed_path.query:
            self._handle_get_prediction(parsed_path.query)
        else:
            self._handle_root()
    
    def do_POST(self):
        """Handle POST requests for predictions"""
        if self.path == '/predict':
            self._handle_post_prediction()
        else:
            self._send_error_response(404, "Endpoint not found")
    
    def _handle_health_check(self):
        """Health check endpoint"""
        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model is not None
        }
        self._send_json_response(response)
    
    def _handle_info(self):
        """Model information endpoint"""
        response = {
            "model_type": "RandomForestClassifier",
            "purpose": "Diabetes Prediction",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "/health": "Health check",
                "/info": "Model information",
                "/predict": "Make predictions (GET with query params or POST with JSON)",
                "/": "This information"
            },
            "model_loaded": self.model is not None
        }
        self._send_json_response(response)
    
    def _handle_root(self):
        """Root endpoint with usage information"""
        html_response = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diabetes Prediction Model</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .example { background: #e8f4fd; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🩺 Diabetes Prediction Model API</h1>
            <p>MLflow-powered Random Forest model for diabetes prediction</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check
            </div>
            <div class="endpoint">
                <strong>GET /info</strong> - Model information
            </div>
            <div class="endpoint">
                <strong>GET|POST /predict</strong> - Make predictions
            </div>
            
            <h2>Example Usage:</h2>
            <div class="example">
                <strong>GET Request:</strong><br>
                <code>/predict?HighBP=1&HighChol=0&CholCheck=1&BMI=25.5&Smoker=0&Stroke=0&HeartDiseaseorAttack=0&PhysActivity=1&Fruits=1&Veggies=1&HvyAlcoholConsump=0&AnyHealthcare=1&NoDocbcCost=0&GenHlth=2&MentHlth=5&PhysHlth=0&DiffWalk=0&Sex=1&Age=8&Education=6&Income=7</code>
            </div>
            <div class="example">
                <strong>POST Request:</strong><br>
                <code>{"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 25.5, ...}</code>
            </div>
            
            <p><em>Status: {'✅ Model Loaded' if self.model else '❌ Model Not Loaded'}</em></p>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_response.encode())
    
    def _handle_get_prediction(self, query_string):
        """Handle GET prediction with query parameters"""
        try:
            # Parse query parameters
            params = parse_qs(query_string)
            
            # Convert to single values (parse_qs returns lists)
            features = {}
            for key, value_list in params.items():
                if value_list:
                    try:
                        features[key] = float(value_list[0])
                    except ValueError:
                        features[key] = value_list[0]
            
            # Make prediction
            prediction_result = self._make_prediction(features)
            self._send_json_response(prediction_result)
            
        except Exception as e:
            logger.error(f"Error in GET prediction: {e}")
            self._send_error_response(400, f"Prediction error: {str(e)}")
    
    def _handle_post_prediction(self):
        """Handle POST prediction with JSON data"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            features = json.loads(post_data.decode('utf-8'))
            
            # Make prediction
            prediction_result = self._make_prediction(features)
            self._send_json_response(prediction_result)
            
        except json.JSONDecodeError:
            self._send_error_response(400, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error in POST prediction: {e}")
            self._send_error_response(400, f"Prediction error: {str(e)}")
    
    def _make_prediction(self, features):
        """Make prediction using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Expected feature names (adjust based on your dataset)
        expected_features = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        
        # Prepare feature vector
        feature_vector = []
        missing_features = []
        
        for feature_name in expected_features:
            if feature_name in features:
                feature_vector.append(float(features[feature_name]))
            else:
                missing_features.append(feature_name)
                feature_vector.append(0.0)  # Default value
        
        # Convert to numpy array
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]
        
        # Prepare response
        result = {
            "prediction": int(prediction),
            "prediction_label": "Diabetes" if prediction == 1 else "No Diabetes",
            "probability": {
                "no_diabetes": float(prediction_proba[0]),
                "diabetes": float(prediction_proba[1])
            },
            "confidence": float(max(prediction_proba)),
            "features_used": len(expected_features),
            "missing_features": missing_features,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        json_data = json.dumps(data, indent=2)
        self.wfile.write(json_data.encode())
    
    def _send_error_response(self, status_code, message):
        """Send error response"""
        error_data = {
            "error": True,
            "status_code": status_code,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self._send_json_response(error_data, status_code)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")

def load_model():
    """Load the trained model"""
    model_paths = [
        "saved_models/",
        "model/",
        "/app/saved_models/",
        "/app/model/"
    ]
    
    model = None
    model_path_used = None
    
    # Try to find and load model
    for base_path in model_paths:
        if os.path.exists(base_path):
            # Look for .pkl files
            for file in os.listdir(base_path):
                if file.endswith('.pkl'):
                    full_path = os.path.join(base_path, file)
                    try:
                        logger.info(f"Attempting to load model from: {full_path}")
                        model = joblib.load(full_path)
                        model_path_used = full_path
                        logger.info(f"✅ Model loaded successfully from: {full_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {full_path}: {e}")
                        continue
            
            if model is not None:
                break
    
    if model is None:
        logger.warning("⚠️  No model could be loaded. Service will run but predictions will fail.")
        logger.info("Available files:")
        for root, dirs, files in os.walk("/app"):
            for file in files:
                if file.endswith('.pkl'):
                    logger.info(f"  Found .pkl file: {os.path.join(root, file)}")
    
    return model, model_path_used

def create_handler_class(model):
    """Create handler class with model"""
    def handler(*args, **kwargs):
        DiabetesPredictionHandler(*args, model=model, **kwargs)
    return handler

def main():
    """Main function to start the prediction server"""
    logger.info("🚀 Starting Diabetes Prediction Model Server...")
    
    # Load model
    model, model_path = load_model()
    
    if model is not None:
        logger.info(f"✅ Model loaded from: {model_path}")
        logger.info(f"📊 Model type: {type(model).__name__}")
    else:
        logger.warning("⚠️  No model loaded - predictions will not work")
    
    # Create server
    port = int(os.environ.get('PORT', 8080))
    server_address = ('', port)
    
    # Create handler class with model
    handler_class = lambda *args, **kwargs: DiabetesPredictionHandler(*args, model=model, **kwargs)
    
    # Start server
    httpd = HTTPServer(server_address, handler_class)
    
    logger.info(f"🌐 Server running on port {port}")
    logger.info(f"📋 Endpoints available:")
    logger.info(f"   http://localhost:{port}/        - Info page")
    logger.info(f"   http://localhost:{port}/health  - Health check")
    logger.info(f"   http://localhost:{port}/info    - Model info")
    logger.info(f"   http://localhost:{port}/predict - Predictions")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Server shutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    main()