import warnings

warnings.filterwarnings("ignore")

import gzip
import io
import os
import pickle
from typing import Optional

import pandas as pd

from src.features.build_features import ALL_FEATURES

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_model(filepath: str):
    with gzip.open(filepath, "rb") as f:
        clf = pickle.Unpickler(f).load()
    return clf


def _load_dataframe_from_bytes(
    file_content: bytes, filename: Optional[str]
) -> pd.DataFrame:
    name = (filename or "").lower()
    buffer = io.BytesIO(file_content)

    if name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(buffer)
    else:
        df = pd.read_excel(buffer)
    return df


def main(
    file_content: Optional[bytes] = None, filename: Optional[str] = None
) -> dict:
    if file_content:
        df = _load_dataframe_from_bytes(file_content, filename)
    else:
        fallback_path = os.path.join(ROOT_DIR, "data", "processed", "bank_test_sample.csv")
        df = pd.read_csv(fallback_path)

    X = df[ALL_FEATURES]

    model_path = os.path.join(ROOT_DIR, "models", "bank_xgb.pkl.gz")
    pipeline = load_model(model_path)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    return {
        "predictions": [int(p) for p in y_pred],
        "probabilities": [round(float(p), 4) for p in y_proba],
    }


if __name__ == "__main__":
    preds = main()
    print(f"Generated {len(preds)} predictions")
