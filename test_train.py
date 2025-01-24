import os
import joblib
import pytest
from sklearn.linear_model import LinearRegression

# Test if the model file exists
def test_model_exists():
    model_file = "model.pkl"
    assert os.path.exists(model_file), f"Model file '{model_file}' not found"

# Test that the model can make accurate predictions
def test_model_prediction():
    # Load the trained model
    model = joblib.load("model.pkl")

    # Test data
    X_test = [[3]]  # We are testing with a value of 3
    expected_prediction = 3

    # Predict using the model
    predicted = model.predict(X_test)[0]

    # Since we are doing simple linear regression with y=x, the prediction should be 3
    assert abs(predicted - expected_prediction) < 1e-2, f"Expected prediction {expected_prediction}, but got {predicted}"

# Run the tests
if __name__ == "__main__":
    pytest.main()
