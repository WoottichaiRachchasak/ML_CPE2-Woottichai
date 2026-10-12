from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "WineQT.csv" 

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    X_raw = df[FEATURES].to_numpy(dtype="float32")  
    X = StandardScaler().fit_transform(X_raw).astype("float32")
    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}