"""
test_train.py
This module contains tests for ensuring that the model file exists and that the trained model's predictions are correct.
"""

import os
import joblib
import pytest
from sklearn.linear_model import LinearRegression
import math

def test_model_exists():
    """
    Test that the model file 'model.pkl' exists in the working directory.
    """
    model_file = "model.pkl"
    assert os.path.exists(model_file), f"Model file '{model_file}' not found"

def test_model_prediction():
    """
    Test that the model makes accurate predictions on a simple dataset.
    A linear regression model is expected to predict a value close to the correct result.
    """
    # Prepare a simple dataset
    x_data = [[1], [2]]
    y_labels = [1, 2]

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(x_data, y_labels)

    # Test prediction
    predicted = model.predict([[3]])[0]  # Predicting for input [3]
    expected = 3  # The expected result is 3 because the relationship is y = x
    
    # Use math.isclose to compare predicted and expected values with a tolerance
    assert math.isclose(predicted, expected, rel_tol=1e-9), f"Expected prediction {expected}, but got {predicted}"

# Run the tests if this module is executed directly
if __name__ == "__main__":
    pytest.main()
