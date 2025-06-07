import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse

def split_data(input_file, output_dir, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except the last one (features)
    y = df.iloc[:, -1]   # Last column (target)

    # Drop non-numeric columns from X (e.g., date/time columns)
    X = X.select_dtypes(include=['float64', 'int64'])

    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save the split datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Data split completed. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    
    args = parser.parse_args()
    split_data(args.input, args.output, args.test_size, args.random_state)
