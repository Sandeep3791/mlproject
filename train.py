from sklearn.linear_model import LinearRegression
import joblib

# Define the main function
def train_and_save_model():
    # Dummy dataset
    X = [[1], [2], [3], [4], [5]]  # Features
    y = [1, 2, 3, 4, 5]            # Labels

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as {model_filename}")

# Execute the main function
if __name__ == "__main__":
    train_and_save_model()
