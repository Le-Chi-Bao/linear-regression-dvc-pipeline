import numpy as np
import joblib
from utils import predict, compute_gradient, update_weights
import yaml

def train_linear_model(X_train, y_train):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
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
        'feature': 'TV',
        'performance': {'final_loss': float(avg_loss)}
    }
    
    joblib.dump(model_data, 'models/linear_model.pkl')
    print(f"Model trained! Final: w={float(w):.4f}, b={float(b):.4f}")
    return w, b, loss_history

if __name__ == "__main__":
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    train_linear_model(X_train, y_train)


