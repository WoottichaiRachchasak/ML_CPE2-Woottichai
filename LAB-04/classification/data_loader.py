"""
Read CSV (Wine Quality Dataset)
make Scaling for KNN Classification
split data: train / validation / test
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ชี้ไปที่ไฟล์ WineQT.csv
CSV_PATH = Path(__file__).resolve().parent.parent / "WineQT.csv"

# คอลัมน์เป้าหมายที่จะใช้ทำนาย (ระดับคุณภาพไวน์)
TARGET = "quality"

# คอลัมน์ตัวเลขสารเคมีทั้ง 11 ตัว
NUMERIC_FEATURES = [
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

# ---------------------------------------------------------------------------
def load_data(test_size=0.2, seed=42):

    # step 1 : read CSV
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()            

    # step 2 : ดึง features ตัวเลขและเป้าหมาย target
    X = df[NUMERIC_FEATURES].to_numpy(dtype="float32")

    # แปลงระดับคุณภาพไวน์ให้เป็น Label ตั้งแต่ 0 เป็นต้นไป
    class_names = [str(c) for c in sorted(df[TARGET].unique())]
    y = df[TARGET].map({int(name): i for i, name in enumerate(class_names)}).to_numpy(dtype="int32")

    # step 3 : split data เป็น train 60% / validation 20% / test 20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp)

    # step 4 : Scaling 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "feature_names": NUMERIC_FEATURES,
        "n_rows": len(df),
    }

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("train :", data["X_train"].shape)
    print("val   :", data["X_val"].shape)
    print("test  :", data["X_test"].shape)
    print("คลาส   :", data["class_names"])