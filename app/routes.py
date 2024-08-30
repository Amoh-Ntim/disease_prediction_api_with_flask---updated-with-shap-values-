from flask import Blueprint, request, jsonify
from .predict import predict_diabetes, predict_heart_disease, predict_kidney_disease, predict_liver_disease
#from collections import OrderedDict
import json
from flask import Response

api = Blueprint('api', __name__)



@api.route('/predict_diabetes', methods=['POST'])
def diabetes_prediction():
    data = request.json
    
    # Call the function and get the dictionary of results
    result = predict_diabetes(data)
    
    # Extract the values from the dictionary
    prediction = result['predicted_class']
    probability = result['probability']
    explanation = result['explanation']
    shap_values = result['shap_values']
    
    # Manually construct the JSON response as a string to maintain order
    response_data = {
        "prediction": [prediction],
        "probability": [probability],
        "explanation": explanation,
        "shap_values": shap_values
    }
    
    response_json = json.dumps(response_data)
    
    return Response(response_json, mimetype='application/json')



@api.route('/predict_heart_disease', methods=['POST'])
def heart_disease_prediction():
    data = request.json
    
    # Call the function and get the dictionary of results
    result = predict_heart_disease(data)
    
    # Extract the values from the dictionary
    prediction = result['predicted_class']
    probability = result['probability']
    explanation = result['explanation']
    shap_values = result['shap_values']
    
    # Manually construct the JSON response as a string to maintain order
    response_data = {
        "prediction": [prediction],
        "probability": [probability],
        "explanation": explanation,
        "shap_values": shap_values
    }
    
    response_json = json.dumps(response_data)
    
    return Response(response_json, mimetype='application/json')


@api.route('/predict_kidney_disease', methods=['POST'])
def kidney_disease_prediction():
    try:
        data = request.json
        result = predict_kidney_disease(data)
        
        # Extract the values from the result dictionary
        prediction = result['predicted_class']
        probability = result['probability']
        explanation = result['explanation']
        
        # Manually construct the JSON response as a string to maintain order
        response_data = {
            "prediction": [prediction],
            "probability": [probability],
            "explanation": explanation
        }
        
        response_json = json.dumps(response_data)
        
        return Response(response_json, mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# new trial code
@api.route('/predict_liver_disease', methods=['POST'])
def liver_disease_prediction():
    data = request.json
    
    # Call the function and get the dictionary of results
    result = predict_liver_disease(data)
    
    # Extract the values from the dictionary
    prediction = result['predicted_class']
    probability = result['probability']
    explanation = result['explanation']
    shap_values = result['shap_values']
    
    # Manually construct the JSON response as a string to maintain order
    response_data = {
        "prediction": [prediction],
        "probability": [probability],
        "explanation": explanation,
        "shap_values": shap_values
    }
    
    response_json = json.dumps(response_data)
    
    return Response(response_json, mimetype='application/json')


