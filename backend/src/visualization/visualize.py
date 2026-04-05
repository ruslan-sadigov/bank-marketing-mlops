"""
Generate model evaluation reports and figures.

Run from backend/:
    uv run python -m src.visualization.visualize

Outputs:
    reports/figures/class_distribution.png
    reports/figures/feature_importance.png
    reports/figures/roc_curve.png
    reports/figures/confusion_matrix.png
    reports/figures/categorical_associations.png
    reports/figures/numeric_correlations.png
    reports/figures/classification_report.png
"""

import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.features.build_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")


def _save(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    phi2corr = max(0.0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def plot_class_distribution(df):
    counts = df["y_numeric"].value_counts().sort_index()
    plot_df = pd.DataFrame({"label": ["No (0)", "Yes (1)"], "count": counts.values})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Target Class Distribution")

    sns.barplot(data=plot_df, x="label", y="count", ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")

    axes[1].pie(
        counts.values,
        labels=[
            f"No (0)\n{counts[0] / counts.sum() * 100:.1f}%",
            f"Yes (1)\n{counts[1] / counts.sum() * 100:.1f}%",
        ],
        autopct="%1.1f%%",
    )

    fig.tight_layout()
    _save(fig, "class_distribution.png")


def plot_feature_importance(pipeline):
    importances = pipeline.named_steps["model"].feature_importances_
    # ColumnTransformer output order: CATEGORICAL_FEATURES (encoded) then NUMERICAL_FEATURES (passthrough)
    feature_names = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", orient="h", ax=ax)
    ax.set_title("XGBoost Feature Importance")

    fig.tight_layout()
    _save(fig, "feature_importance.png")


def plot_roc_curve(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.lineplot(
        data=roc_df, x="fpr", y="tpr", ax=ax, label=f"XGBoost (AUC = {auc:.4f})"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()
    _save(fig, "roc_curve.png")
    return auc


def plot_confusion_matrix(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No (0)", "Yes (1)"]
    ).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    _save(fig, "confusion_matrix.png")
    return y_pred


def plot_categorical_associations(df):
    cv_values = {
        col: cramers_v(df[col], df["y_numeric"]) for col in CATEGORICAL_FEATURES
    }
    cv_df = pd.DataFrame({"feature": cv_values.keys(), "cramers_v": cv_values.values()})
    cv_df = cv_df.sort_values("cramers_v")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=cv_df, x="cramers_v", y="feature", orient="h", ax=ax)
    ax.set_xlabel("Cramér's V")
    ax.set_title("Categorical Feature Association with Target")

    fig.tight_layout()
    _save(fig, "categorical_associations.png")


def plot_numeric_correlations(df):
    corrs = {col: spearmanr(df[col], df["y_numeric"])[0] for col in NUMERICAL_FEATURES}
    corr_df = pd.DataFrame({"feature": corrs.keys(), "correlation": corrs.values()})
    corr_df = corr_df.sort_values("correlation")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=corr_df, x="correlation", y="feature", orient="h", ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman Correlation with Target")
    ax.set_title("Numerical Feature Correlation with Target")

    fig.tight_layout()
    _save(fig, "numeric_correlations.png")


def save_classification_report(y_test, y_pred, auc):
    report_dict = classification_report(
        y_test, y_pred, target_names=["No (0)", "Yes (1)"], output_dict=True
    )
    assert isinstance(report_dict, dict)

    rows = ["No (0)", "Yes (1)", "macro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score"]
    data = {row: {m: report_dict[row][m] for m in metrics} for row in rows}  # type: ignore[index]
    df_report = pd.DataFrame(data).T

    fig, axes = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [4, 1]}
    )
    fig.suptitle(
        "XGBoost Classification Report — Bank Marketing", fontsize=13, fontweight="bold"
    )

    sns.heatmap(
        df_report,
        annot=True,
        fmt=".3f",
        vmin=0,
        vmax=1,
        cmap="Blues",
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title("Precision / Recall / F1-Score per Class", fontsize=10)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].tick_params(axis="y", rotation=0)

    axes[1].axis("off")
    summary = (
        f"Accuracy: {report_dict['accuracy']:.3f}     "
        f"ROC-AUC: {auc:.4f}     "
        f"Support (total): {int(report_dict['weighted avg']['support'])}"  # type: ignore[index]
    )
    axes[1].text(0.5, 0.5, summary, ha="center", va="center", fontsize=11)

    fig.tight_layout()
    _save(fig, "classification_report.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "bank_train.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "processed", "bank_test.csv"))
    X_test, y_test = test_df[ALL_FEATURES], test_df["y_numeric"]

    with gzip.open(os.path.join(ROOT_DIR, "models", "bank_xgb.pkl.gz"), "rb") as f:
        pipeline = pickle.loads(f.read())

    print("Generating figures...")
    plot_class_distribution(df)
    plot_feature_importance(pipeline)
    auc = plot_roc_curve(pipeline, X_test, y_test)
    y_pred = plot_confusion_matrix(pipeline, X_test, y_test)
    plot_categorical_associations(df)
    plot_numeric_correlations(df)
    save_classification_report(y_test, y_pred, auc)

    print("\nDone. All outputs saved to reports/")


if __name__ == "__main__":
    main()
