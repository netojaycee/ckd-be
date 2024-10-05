import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Load models and scaler
lgbm = joblib.load("model/lgbm_model.pkl")
xgboost = joblib.load("model/xgboost_model.pkl")
decision_tree = joblib.load("model/decision_tree_model.pkl")
scaler = joblib.load("model/scaler.pkl")

X_columns = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", 
    "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"
]

def preprocess_input(data):
    df = pd.DataFrame([data])
    for x in ["rc", "wc", "pcv"]:
        df[x] = df[x].astype(float)

    # Fill missing values
    for x in [
        "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "rc", "wc", "pcv"
    ]:
        df[x].fillna(df[x].mean(), inplace=True)

    # Replace categorical values
    df[["htn", "dm", "cad", "pe", "ane"]] = df[["htn", "dm", "cad", "pe", "ane"]].replace({"yes": 1, "no": 0})
    df[["rbc", "pc"]] = df[["rbc", "pc"]].replace({"abnormal": 1, "normal": 0})
    df[["pcc", "ba"]] = df[["pcc", "ba"]].replace({"present": 1, "notpresent": 0})
    df[["appet"]] = df[["appet"]].replace({"good": 1, "poor": 0, "no": np.nan})

    # One-hot encode
    df, _ = one_hot_encoder(df, nan_as_category=True)

    # Align columns
    missing_cols = set(X_columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0
    df = df[X_columns]

    # Scale data
    df_scaled = scaler.transform(df)
    return df_scaled

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# New data for prediction
new_data = {
    "age": 52.0,
    "bp": 80.0,
    "sg": 1.02,
    "al": 0.0,
    "su": 0.0,
    "rbc": "normal",
    "pc": "normal",
    "pcc": "notpresent",
    "ba": "notpresent",
    "bgr": 125.0,
    "bu": 22.0,
    "sc": 1.2,
    "sod": 139.0,
    "pot": 4.6,
    "hemo": 16.5,
    "pcv": 43,
    "wc": 4700,
    "rc": 4.6,
    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
}

# Preprocess the new data
preprocessed_new_data = preprocess_input(new_data)

# Predict using the models
lgbm_pred = lgbm.predict(preprocessed_new_data)
xgboost_pred = xgboost.predict(preprocessed_new_data)
decision_tree_pred = decision_tree.predict(preprocessed_new_data)

print(f"LightGBM Prediction: {int(lgbm_pred[0])}")
print(f"XGBOOST Prediction: {int(xgboost_pred[0])}")
print(f"Decision Tree Prediction: {int(decision_tree_pred[0])}")
