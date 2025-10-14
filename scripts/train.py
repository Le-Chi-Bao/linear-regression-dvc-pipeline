import numpy as np
import joblib
from utils import predict, compute_gradient, update_weights
import yaml

def train_linear_model(X_train, y_train):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    version = params['data']['version']
    feature = params['data']['feature'] 
    epochs = params['train']['epochs']
    lr = params['train']['lr']
    
    print(f" Training với: epochs={epochs}, lr={lr}")
    # NORMALIZE DATA để tránh gradient explosion
    X_mean, X_std = np.mean(X_train), np.std(X_train)
    y_mean, y_std = np.mean(y_train), np.std(y_train)

    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std

    w, b = 0.0, 0.0                  # khởi tạo các tham số 
    N = len(X_train)
    
    loss_history = []                # Lưu lại loss trong quá trình train
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(N):
            y_hat = predict(X_train[i], w, b)
            loss = (y_hat - y_train[i]) ** 2
            total_loss += loss
            
            d_w, d_b = compute_gradient(X_train[i], y_hat, y_train[i])
            w, b = update_weights(w, b, d_w, d_b, lr)
        
        avg_loss = total_loss / N                # trung bình loss cho mỗi epoch
        loss_history.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss={float(avg_loss):.4f}")
    
    # Lưu model
    model_data = {
        'w': float(w),
        'b': float(b),
        'feature': feature,
        'performance': {'final_loss': float(avg_loss)},
        'normalization': { 
            'X_mean': float(X_mean),
            'X_std': float(X_std), 
            'y_mean': float(y_mean),
            'y_std': float(y_std)
        }
    }

    model_path = f'models/model_{version}.pkl'
    joblib.dump(model_data, model_path)
    print(f"Model trained! Final: w={float(w):.4f}, b={float(b):.4f}")
    print(f"✅ Model saved: {model_path}")
    return w, b, loss_history

if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    version = params['data']['version']
    
    print(f"🚀 Training with data version: {version}")
    
    X_train = np.load(f'data/processed/X_train_{version}.npy')
    y_train = np.load(f'data/processed/y_train_{version}.npy')
    train_linear_model(X_train, y_train)

