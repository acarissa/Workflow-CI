import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
import sys
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    
    # Akses train dan test data
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laptop_price_preprocessing")
    
    train_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_path, "train.csv")
    test_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(base_path, "test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    # Splitting ulang
    X_train, y_train = train_data.drop("Price_Category", axis=1), train_data["Price_Category"]
    X_test, y_test = test_data.drop("Price_Category", axis=1), test_data["Price_Category"]

    input_example = X_train[0:5]

    with mlflow.start_run():
        # Log parameters
        mlflow.autolog()
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )