from flask import Flask, render_template, request
import numpy as np

from src.data_loader import load_data
from src.features import create_features
from src.predict import make_prediction, load_models

# ✅ 1. CREATE APP (MUST BE FIRST)
app = Flask(__name__)

# ✅ 2. LOAD DATA + MODELS
df = load_data("data/energy.csv")
df = create_features(df)

X = df.drop(columns=['Energy'])
y = df['Energy']

lr, mlp = load_models()

# ✅ 3. HOME ROUTE (ADD THIS HERE)
@app.route('/')
def home():

    preds = mlp.predict(X)

    return render_template("index.html",
        pred_lr=0,
        pred_mlp=0,

        time_series=y.values[-50:].tolist(),
        predicted=preds[-50:].tolist(),

        hourly=df.groupby('hour')['Energy'].mean().tolist(),
        weekly=df.groupby('day')['Energy'].mean().tolist(),

        error=(y - preds).values[-50:].tolist(),

        mae_lr=round(np.mean(abs(y - lr.predict(X))),2),
        mae_mlp=round(np.mean(abs(y - preds)),2),

        avg=round(np.mean(y),2),
        max_val=round(np.max(y),2),
        min_val=round(np.min(y),2)
    )

# ✅ 4. PREDICT ROUTE (ADD BELOW HOME)
@app.route('/predict', methods=['POST'])
def predict():

    hour = int(request.form['hour'])
    day = int(request.form['day'])

    last_row = X.iloc[-1].copy()

    pred_lr, pred_mlp = make_prediction(hour, day, last_row)

    # 🔥 Add new predicted value
    new_series = y.values.tolist()
    new_series.append(pred_mlp)

    preds = mlp.predict(X).tolist()
    preds.append(pred_mlp)

    return render_template("index.html",
        pred_lr=pred_lr,
        pred_mlp=pred_mlp,

        time_series=new_series[-50:],
        predicted=preds[-50:],

        hourly=df.groupby('hour')['Energy'].mean().tolist(),
        weekly=df.groupby('day')['Energy'].mean().tolist(),

        error=(y - mlp.predict(X)).values[-50:].tolist(),

        mae_lr=round(np.mean(abs(y - lr.predict(X))),2),
        mae_mlp=round(np.mean(abs(y - mlp.predict(X))),2),

        avg=round(np.mean(new_series),2),
        max_val=round(np.max(new_series),2),
        min_val=round(np.min(new_series),2)
    )

# ✅ 5. RUN APP (LAST)
if __name__ == "__main__":
    app.run(debug=True)