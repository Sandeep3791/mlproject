"""Script to train and save the model."""

from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model():
    """Train the linear regression model and save it as a file."""
    # Dummy dataset
    x = [[1], [2], [3], [4], [5]]  # Features
    y = [1, 2, 3, 4, 5]            # Labels

    # Initialize and train the model
    model = LinearRegression()
    model.fit(x, y)

    # Save the trained model to a file
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as {model_filename}")

# Execute the main function
if __name__ == "__main__":
    train_and_save_model()
