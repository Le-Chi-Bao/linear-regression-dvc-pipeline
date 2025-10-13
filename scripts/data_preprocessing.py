import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import yaml  

def load_and_preprocess_data(data_path):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    df = pd.read_csv(data_path)
    
    X = df[params['data']['feature']].values.reshape(-1, 1)
    y = df[params['data']['target']].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state']
    )

    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test) 
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    print("Data preprocessing completed!")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data('data/raw/advertising.csv')