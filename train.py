"""
This module trains a simple Linear Regression model and saves it to a file.
"""

import pickle
from sklearn.linear_model import LinearRegression


def train_model():
    """
    Trains a Linear Regression model on a simple dataset and saves it as a .pkl file.
    """
    # Dataset
    x_data = [[1], [2], [3], [4], [5]]
    y_labels = [1, 2, 3, 4, 5]

    # Train the model
    model = LinearRegression()
    model.fit(x_data, y_labels)

    # Save the model to a file
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print("Model saved successfully as model.pkl")


if __name__ == "__main__":
    train_model()
