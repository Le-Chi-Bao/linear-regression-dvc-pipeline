def predict(X, w, b):
    return w * X + b

def compute_gradient(X, y_hat, y):
    dw = 2 * X * (y_hat - y)
    db = 2 * (y_hat - y)
    return dw, db

def update_weights(w, b, dw, db, lr):
    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new

def evaluate_model(X_test, y_test, w, b):
    y_pred = w * X_test + b
    test_loss = np.mean((y_pred - y_test) ** 2)
    test_mae = np.mean(np.abs(y_pred - y_test))
    return test_loss, test_mae