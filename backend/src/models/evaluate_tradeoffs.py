import warnings

warnings.filterwarnings("ignore")

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from src.features.build_features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def build_pipeline(scale_pos_weight: float) -> Pipeline:
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
    model = XGBClassifier(
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
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def fit_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]):
    y_train, y_test = train_df["y_numeric"], test_df["y_numeric"]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(train_df[features], y_train)
    proba = pipeline.predict_proba(test_df[features])[:, 1]
    return y_test.to_numpy(), proba


def main():
    train_df = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "bank_train.csv")
    )
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "bank_test.csv"))

    production_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    # Production model (no `duration`) vs. the same model with `duration` added.
    # `duration` is only known after a call, so the second model is a leakage reference.
    y_test, proba = fit_and_score(train_df, test_df, production_features)
    _, proba_leaky = fit_and_score(
        train_df, test_df, production_features + ["duration"]
    )

    print("Model comparison (test set)")
    print("| Features | ROC-AUC | PR-AUC |")
    print("|---|---|---|")
    print(
        f"| Production (without duration) | {roc_auc_score(y_test, proba):.3f} "
        f"| {average_precision_score(y_test, proba):.3f} |"
    )
    print(
        f"| With duration (leakage reference) | {roc_auc_score(y_test, proba_leaky):.3f} "
        f"| {average_precision_score(y_test, proba_leaky):.3f} |"
    )

    print("\nThreshold trade-off, production model (positive class)")
    print("| Threshold | Precision | Recall | Customers flagged |")
    print("|---|---|---|---|")
    for threshold in THRESHOLDS:
        y_pred = (proba >= threshold).astype(int)
        print(
            f"| {threshold:.1f} | {precision_score(y_test, y_pred):.3f} "
            f"| {recall_score(y_test, y_pred):.3f} | {y_pred.mean():.1%} |"
        )

    top_decile = int(len(y_test) * 0.1)
    ranked = y_test[(-proba).argsort()][:top_decile]
    print(
        f"\nTop 10% of customers by score contain {ranked.sum() / y_test.sum():.1%} "
        f"of all subscribers (random baseline: 10.0%)"
    )


if __name__ == "__main__":
    main()
