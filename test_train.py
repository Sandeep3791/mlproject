"""
This module contains unit tests for the Linear Regression model training and prediction.
"""

import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sample_model():
    """
    Fixture to provide a simple Linear Regression model trained on a small dataset.
    """
    x_data = [[1], [2], [3]]
    y_labels = [1, 2, 3]

    model = LinearRegression()
    model.fit(x_data, y_labels)
    return model


def test_model_prediction(sample_model):
    """
    Test that the model makes accurate predictions on a simple dataset.
    """
    predicted = sample_model.predict([[4]])[0]
    expected = 4
    assert predicted == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {predicted}"
