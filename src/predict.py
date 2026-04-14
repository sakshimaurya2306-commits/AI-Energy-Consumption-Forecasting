import joblib
import numpy as np

def load_models():
    lr = joblib.load("models/lr_model.pkl")
    mlp = joblib.load("models/mlp_model.pkl")
    return lr, mlp


def make_prediction(hour, day, last_row):

    lr, mlp = load_models()

    # Copy last known data (important for lag features)
    x = last_row.copy()

    # Update user inputs
    x['hour'] = hour
    x['day'] = day

    # Convert to model format
    x_array = np.array([x.values])

    # Predict
    pred_lr = lr.predict(x_array)[0]
    pred_mlp = mlp.predict(x_array)[0]

    return round(pred_lr, 2), round(pred_mlp, 2)