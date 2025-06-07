import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import os
import argparse

def evaluate_model(X_test_file, y_test_file, model_file, output_dir, metrics_dir):
    """Evaluate trained model and save metrics."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load test data
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file).values.ravel()
    
    # Load trained model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Save metrics
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2)
    }
    
    metrics_file = os.path.join(metrics_dir, 'scores.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    predictions_file = os.path.join(output_dir, 'prediction.csv')
    predictions_df.to_csv(predictions_file, index=False)

    print(f"Model evaluation completed:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Metrics saved to {metrics_file}")
    print(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X-test", required=True, help="Test features file")
    parser.add_argument("--y-test", required=True, help="Test target file")
    parser.add_argument("--model", required=True, help="Trained model file")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    parser.add_argument("--metrics", required=True, help="Metrics output directory")
    
    args = parser.parse_args()
    evaluate_model(args.X_test, args.y_test, args.model, args.output, args.metrics)
