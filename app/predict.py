import os
import joblib
from pycaret.classification import load_model, predict_model
import pandas as pd
import shap  
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
diabetes_model = load_model(os.path.join(base_dir, '../models/diabetes_prediction_model'))
heart_model = load_model(os.path.join(base_dir, '../models/best_heart_disease_model'))
kidney_model = load_model(os.path.join(base_dir, '../models/best_kidney_disease_model'))
liver_model = load_model(os.path.join(base_dir, '../models/best_liver_disease_model'))

# Load training data for SHAP explainers
diabetes_train_data = joblib.load(os.path.join(base_dir, '../data/diabetes_train_data.pkl'))
heart_train_data = joblib.load(os.path.join(base_dir, '../data/heart_train_data.pkl'))
liver_train_data = joblib.load(os.path.join(base_dir, '../data/liver_train_data.pkl'))
kidney_train_data = joblib.load(os.path.join(base_dir, '../data/kidney_train_data.pkl'))


def preprocess_input(data, expected_features):
    """
    Ensures the input data is a pandas DataFrame with numeric types, 
    and contains all the expected features, filling missing ones with default values.
    """
    # Convert input data to a DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all columns are numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Ensure the DataFrame contains all expected features
    df = df.reindex(columns=expected_features, fill_value=0)
    
    # Fill NaNs with 0 or another appropriate default value
    df = df.fillna(0)
    
    return df


def get_shap_values(model, data, train_data):
    explainer = shap.Explainer(model.predict, train_data)
    shap_values = explainer(data)
    return shap_values.values.tolist()


def predict_diabetes(data):
    expected_features = diabetes_train_data.columns.tolist()
    df = preprocess_input(data, expected_features)
    
    # Make predictions using the trained diabetes model
    prediction = predict_model(diabetes_model, data=df)
    
    # Get SHAP values to explain the model's prediction
    shap_values = get_shap_values(diabetes_model, df, diabetes_train_data)
    
    # Extract predicted class (label) and prediction probability (score)
    predicted_class = int(prediction['prediction_label'][0])
    predicted_prob = float(prediction['prediction_score'][0])

    # Map SHAP values to corresponding feature names
    feature_names = df.columns
    shap_values_mapped = dict(zip(feature_names, shap_values[0]))

    # Explanation for the prediction result
    predicted_class_str = "Diabetic" if predicted_class == 1 else "Non Diabetic"
    
    # Explanation of prediction probability
    if predicted_class == 1:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient has diabetes."
    else:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient does not have diabetes."

    return {
        "predicted_class": predicted_class_str,
        "probability": predicted_prob,
        "shap_values": shap_values_mapped,
        "explanation": prob_explanation
    }


def predict_heart_disease(data):
    expected_features = heart_train_data.columns.tolist()
    if 'target' in expected_features:
        expected_features.remove('target')
        
    df = preprocess_input(data, expected_features)
    
    prediction = predict_model(heart_model, data=df)
    
    background_data = heart_train_data.drop(columns=['target']) if 'target' in heart_train_data.columns else heart_train_data
    
    explainer = shap.Explainer(heart_model.predict, background_data)
    shap_values = explainer(df)
    
    feature_names = df.columns
    shap_values_mapped = dict(zip(feature_names, shap_values.values[0]))
    
    predicted_class = int(prediction['prediction_label'][0])
    predicted_prob = float(prediction['prediction_score'][0])

    predicted_class_str = "Heart Disease" if predicted_class == 1 else "No Heart Disease"
    
    if predicted_class == 1:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient has heart disease."
    else:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient does not have heart disease."

    return {
        "predicted_class": predicted_class_str,
        "probability": predicted_prob,
        "shap_values": shap_values_mapped,
        "explanation": prob_explanation
    }


def predict_kidney_disease(data):
    expected_features = ['age', 'bp', 'sg', 'al', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wc', 'htn', 'dm', 'appet', 'rbc', 'ane']
    
    df = preprocess_input(data, expected_features)
    
    prediction = predict_model(kidney_model, data=df)
    
    predicted_class = int(prediction['prediction_label'][0])
    predicted_prob = round(float(prediction['prediction_score'][0]), 4)

    predicted_class_str = "Chronic Kidney Disease (CKD)" if predicted_class == 1 else "No Chronic Kidney Disease (Non-CKD)"
    
    if predicted_class == 1:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient has chronic kidney disease."
    else:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient does not have chronic kidney disease."

    return {
        "predicted_class": predicted_class_str,
        "probability": predicted_prob,
        "explanation": prob_explanation
    }


def predict_liver_disease(data):
    expected_features = liver_train_data.columns.tolist()
    if 'Dataset' in expected_features:
        expected_features.remove('Dataset')
        
    df = preprocess_input(data, expected_features)
    
    prediction = predict_model(liver_model, data=df)
    
    background_data = liver_train_data.drop(columns=['Dataset']) if 'Dataset' in liver_train_data.columns else liver_train_data
    
    explainer = shap.Explainer(liver_model.predict, background_data)
    shap_values = explainer(df)
    
    feature_names = df.columns
    shap_values_mapped = dict(zip(feature_names, shap_values.values[0]))
    
    predicted_class = int(prediction['prediction_label'][0])
    predicted_prob = round(float(predicted_class['prediction_score'][0]), 4)

    predicted_class_str = "Liver Disease" if predicted_class == 1 else "No Liver Disease"
    
    if predicted_class == 1:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient has liver disease."
    else:
        prob_explanation = f"There is a {predicted_prob * 100:.2f}% probability that the patient does not have liver disease."

    return {
        "predicted_class": predicted_class_str,
        "probability": predicted_prob,
        "shap_values": shap_values_mapped,
        "explanation": prob_explanation
    }
