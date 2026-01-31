import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model/trained_model.pkl"

def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def train_model(model_type="linear"):
    df = load_data()

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "linear":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return rmse, r2

def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict(input_data):
    model = load_model()
    return model.predict(input_data)