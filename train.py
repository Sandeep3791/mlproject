from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    x_data = [[1], [2], [3], [4]]
    y_labels = [2, 4, 6, 8]
    model = LinearRegression()
    model.fit(x_data, y_labels)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved successfully as model.pkl")

if __name__ == "__main__":
    train_model()
