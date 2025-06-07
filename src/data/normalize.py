import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse

def normalize_data(train_file, test_file, output_dir):
    """Normalize training and testing data using StandardScaler."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    X_train = pd.read_csv(train_file)
    X_test = pd.read_csv(test_file)
    
    # Initialize and fit the scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save normalized data
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    
    # Save the scaler for later use
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Data normalization completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="Training data file")
    parser.add_argument("--test-file", required=True, help="Test data file")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    normalize_data(args.train_file, args.test_file, args.output)
