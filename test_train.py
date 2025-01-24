"""Test file for training the model."""

import os
from sklearn.linear_model import LinearRegression

def test_model_exists():
    """Test if the model file exists."""
    model_file = "model.pkl"
    assert os.path.exists(model_file), f"Model file '{model_file}' not found"

def test_model_prediction():
    """Test if the model makes accurate predictions."""
    # Prepare a simple dataset
    x = [[1], [2]]
    y = [1, 2]

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(x, y)

    # Test prediction
    predicted = model.predict([[3]])[0]
    expected = 3
    assert predicted == expected, f"Expected prediction {expected}, but got {predicted}"
