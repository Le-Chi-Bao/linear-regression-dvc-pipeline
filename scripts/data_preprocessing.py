import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import yaml  

def load_and_preprocess_data(data_path):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    version = params['data']['version']
    df = pd.read_csv(data_path)
    
    X = df[params['data']['feature']].values.reshape(-1, 1)
    y = df[params['data']['target']].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state']
    )

    np.save(f'data/processed/X_train_{version}.npy', X_train)
    np.save(f'data/processed/X_test_{version}.npy', X_test)
    np.save(f'data/processed/y_train_{version}.npy', y_train)
    np.save(f'data/processed/y_test_{version}.npy', y_test)
    
    print(f"✅ Saved data version: {version}")
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data('data/raw/advertising.csv')

    