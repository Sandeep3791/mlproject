import pytest
from sklearn.linear_model import LinearRegression

def test_model_prediction():
    x_data = [[1], [2]]
    y_labels = [1, 2]

    model = LinearRegression()
    model.fit(x_data, y_labels)

    predicted = model.predict([[3]])[0]
    expected = 3
    assert predicted == pytest.approx(expected, abs=1e-6), f"Expected {expected}, got {predicted}"
