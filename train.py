from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model():
    # Training data
    X = [[1], [2], [3], [4], [5]]
    y = [1, 2, 3, 4, 5]

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as {model_filename}")

if __name__ == "__main__":
    train_and_save_model()
