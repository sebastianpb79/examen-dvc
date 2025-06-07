import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import argparse

def train_model(X_train_file, y_train_file, params_file, output_dir):
    """Train model with best parameters."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file).values.ravel()
    
    # Load best parameters
    with open(params_file, 'rb') as f:
        best_params = pickle.load(f)
    
    # Train model with best parameters
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    # Save trained model
    model_file = os.path.join(output_dir, 'gbr_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model training completed. Model saved to {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X-train", required=True, help="Training features file")
    parser.add_argument("--y-train", required=True, help="Training target file")
    parser.add_argument("--params", required=True, help="Best parameters file")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    train_model(args.X_train, args.y_train, args.params, args.output)
