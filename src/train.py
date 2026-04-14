import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def train_models(X, y):

    lr = LinearRegression()
    mlp = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500)

    lr.fit(X, y)
    mlp.fit(X, y)

    joblib.dump(lr, "models/lr_model.pkl")
    joblib.dump(mlp, "models/mlp_model.pkl")

    print("✅ Models trained and saved")