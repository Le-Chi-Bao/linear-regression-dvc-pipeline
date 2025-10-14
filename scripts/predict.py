import joblib
import numpy as np
import yaml

def predict_sales(feature_value, model_path=None):
    """Dự đoán sales từ feature value (TV, radio, hoặc newspaper) với model univariate"""
    
    # Load params để lấy version
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    version = params['data']['version']
    
    # Nếu không chỉ định model_path, dùng model theo version
    if model_path is None:
        model_path = f'models/model_{version}.pkl'
    
    # Load model
    model_data = joblib.load(model_path)
    
    w = model_data['w']
    b = model_data['b']
    model_feature = model_data['feature'] 
    normalization = model_data.get('normalization', {})
    
    print(f"🧪 Predicting with: {model_path}")
    print(f"   Model: y = {w:.4f}*{model_feature} + {b:.4f}")
    
    # Đảm bảo feature_value là array shape [-1, 1]
    if isinstance(feature_value, (int, float)):
        feature_value = np.array([[feature_value]])  # shape: [1, 1]
    else:
        feature_value = np.array(feature_value).reshape(-1, 1)  # shape: [n, 1]
    
    # NORMALIZE input nếu có normalization parameters
    if normalization:
        X_mean = normalization['X_mean']
        X_std = normalization['X_std']
        y_mean = normalization['y_mean'] 
        y_std = normalization['y_std']
        
        # Normalize input
        feature_normalized = (feature_value - X_mean) / X_std
        
        # Predict với data normalized
        prediction_normalized = w * feature_normalized + b
        
        # Denormalize output
        prediction = prediction_normalized * y_std + y_mean
    else:
        # Predict không normalize
        prediction = w * feature_value + b
        print(" ⚠️  No normalization found - using direct prediction")

    return prediction.flatten()  # Trả về 1D array để dễ sử dụng


import joblib
import numpy as np
import yaml

def predict_sales(feature_value, model_path=None):
    """Dự đoán sales từ feature value với model univariate"""
    
    # Load params để lấy version
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    version = params['data']['version']
    
    # Nếu không chỉ định model_path, dùng model theo version
    if model_path is None:
        model_path = f'models/model_{version}.pkl'
    
    # Load model
    model_data = joblib.load(model_path)
    
    w = model_data['w']
    b = model_data['b']
    model_feature = model_data['feature']
    normalization = model_data.get('normalization', {})
    
    print(f"🧪 Predicting with: {model_path}")
    print(f"   Model: y = {w:.4f}*{model_feature} + {b:.4f}")
    
    # Đảm bảo feature_value là array shape [-1, 1]
    if isinstance(feature_value, (int, float)):
        feature_value = np.array([[feature_value]])  # shape: [1, 1]
    else:
        feature_value = np.array(feature_value).reshape(-1, 1)  # shape: [n, 1]
    
    # Xử lý normalization
    if normalization and all(key in normalization for key in ['X_mean', 'X_std', 'y_mean', 'y_std']):
        X_mean = normalization['X_mean']
        X_std = normalization['X_std']
        y_mean = normalization['y_mean'] 
        y_std = normalization['y_std']
        
        # Normalize input
        feature_normalized = (feature_value - X_mean) / X_std
        prediction_normalized = w * feature_normalized + b
        prediction = prediction_normalized * y_std + y_mean
        print("   🔄 Using normalized prediction")
    else:
        # Predict không normalize
        prediction = w * feature_value + b
        print("   ⚠️  No normalization found - using direct prediction")
    
    return prediction.flatten()  # Trả về 1D array

if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    predict_params = params['predict']
    values = predict_params['values']
    model_path = predict_params['model_path']
    
    print("🎯 SALES PREDICTION FROM PARAMS.YAML")
    
    # Predict tất cả values được chỉ định
    predictions = predict_sales(values, model_path)
    
    print("\n PREDICTION RESULTS:")
    for value, pred in zip(values, predictions):
        print(f"   ${value}: ${pred:.2f} sales")

