import joblib
import numpy as np

def predict_sales(tv_budget, model_path='models/linear_model.pkl'):
    """Dự đoán sales từ TV budget"""
    model_data = joblib.load(model_path)
    w = model_data['w']
    b = model_data['b']
    
    prediction = w * tv_budget + b
    return prediction

if __name__ == "__main__":
    result = predict_sales(200)
    print(f" Prediction for TV $200: ${result:.2f} sales")

