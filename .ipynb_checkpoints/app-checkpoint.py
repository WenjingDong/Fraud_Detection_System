from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from config import pipeline_ONEHOT_ENCODED_FEATURES, pipeline_LABEL_ENCODED_FEATURES, SCALED_FEATURES, SELECTED_FEATURES
from feature_transformer import scale_features, transform_features, FeatureTransformer, FeatureScaler, FeatureSelector

app = Flask(__name__)

# Load your trained fraud detection model
pipeline = joblib.load('fraud_detection_model.pkl')  # Adjust the path to your model file


# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Make sure the HTML file is in a 'templates' folder


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()  # Expecting JSON data
        # print("Received data:", data) # Debug 
        if not data:
            return jsonify({"error": "Invalid input"}), 400
        
        # Extract features from the input data
        # print(type(data)) # Debug
        features = pd.DataFrame(data, index=[0]) 
        # print("Passed features is:")
        # print(features) # Debug
        # print(features.dtypes)
        
        
        # Debug pipeline step by step
        # print("Debug transformer")
        # features_copy = features.copy()
        # features_transformed = pipeline.named_steps['featureTransformer'].transform(features_copy)
        # print("After first transformation is:")
        # print(features_transformed[['cc_num_encoded', 'zip_encoded']].T)
        # features_transformed = pipeline.named_steps['featureScaler'].transform(features_transformed)
        # features_transformed = pipeline.named_steps['featureSelector'].transform(features_transformed)
        # print("Transformed features is:")
        # print(features_transformed)
        # prediction = pipeline.named_steps['classifier'].predict(features_transformed)
        # print("prediction:", prediction)
        
        # Make prediction
        prediction = pipeline.predict(features)
        # print("Prediction by the pipeline", prediction)


        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()[0]})
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 400


# Define a health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True)
