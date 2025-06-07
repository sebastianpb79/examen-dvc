import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os
import argparse

def perform_gridsearch(X_train_file, y_train_file, output_dir):
    """Perform GridSearch to find best parameters."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file).values.ravel()
    
    # Define the model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform GridSearch
    print("Starting GridSearch...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Save best parameters
    best_params_file = os.path.join(output_dir, 'best_params.pkl')
    with open(best_params_file, 'wb') as f:
        pickle.dump(grid_search.best_params_, f)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
    print(f"Best parameters saved to {best_params_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X-train", required=True, help="Training features file")
    parser.add_argument("--y-train", required=True, help="Training target file")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    perform_gridsearch(args.X_train, args.y_train, args.output)
