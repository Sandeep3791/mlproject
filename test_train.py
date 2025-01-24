import os
import joblib
from sklearn.linear_model import LinearRegression

# Test if the model file exists
def test_model_exists():
    model_file = "model.pkl"
    # Ensure the model is saved if not already present
    if not os.path.exists(model_file):
        # Train and save the model
        model = LinearRegression()
        model.fit([[1], [2]], [1, 2])
        joblib.dump(model, model_file)
    
    # Test if the model file exists
    assert os.path.exists(model_file), f"Model file '{model_file}' not found"

# Test if the model makes accurate predictions
def test_model_prediction():
    # Prepare a simple dataset
    X = [[1], [2]]
    y = [1, 2]

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Test prediction
    predicted = model.predict([[3]])[0]
    expected = 3
    
    # Use np.isclose to handle small floating-point differences
    assert abs(predicted - expected) < 1e-5, f"Expected prediction {expected}, but got {predicted}"

# Run the tests
if __name__ == "__main__":
    try:
        test_model_exists()
        print("Test 'test_model_exists' passed.")
    except AssertionError as e:
        print(f"Test 'test_model_exists' failed: {e}")

    try:
        test_model_prediction()
        print("Test 'test_model_prediction' passed.")
    except AssertionError as e:
        print(f"Test 'test_model_prediction' failed: {e}")
