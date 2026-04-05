import warnings

warnings.filterwarnings("ignore")

import gzip
import os
import pickle
import pickletools

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.features.build_features import ALL_FEATURES, CATEGORICAL_FEATURES

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def save_model(filename: str, model: object):
    file_path = os.path.join(ROOT_DIR, "models", filename)
    with gzip.open(file_path, "wb") as f:
        pickled = pickle.dumps(model)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


def main():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "bank_train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "bank_test.csv"))

    X_train, y_train = train_df[ALL_FEATURES], train_df["y_numeric"]
    X_test, y_test = test_df[ALL_FEATURES], test_df["y_numeric"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=425,
                    learning_rate=0.0122498,
                    max_depth=12,
                    reg_lambda=8.681178,
                    subsample=0.788238,
                    colsample_bytree=0.596907,
                    min_child_weight=9,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbosity=0,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    save_model("bank_xgb.pkl.gz", pipeline)
    print("Model saved to: bank_xgb.pkl.gz")


if __name__ == "__main__":
    main()
