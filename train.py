"""
train.py
This module is responsible for training a linear regression model on a dummy dataset and saving it to a file.
"""

from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model():
    """
    Train a linear regression model on a simple dataset and save it as a .pkl file.
    The dataset consists of 5 data points with a simple linear relationship.
    """
    # Dummy dataset
    x_data = [[1], [2], [3], [4], [5]]  # Features
    y_labels = [1, 2, 3, 4, 5]          # Labels

    # Initialize and train the model
    model = LinearRegression()
    model.fit(x_data, y_labels)

    # Save the trained model to a file
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as {model_filename}")

# Run the training function if this module is run directly
if __name__ == "__main__":
    train_and_save_model()
