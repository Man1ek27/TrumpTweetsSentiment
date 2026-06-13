"""SHAP explainability utilities for sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def _slice_for_class(shap_values: shap.Explanation, class_index: int) -> shap.Explanation:
    """Slice SHAP explanation for a selected class in multiclass setups."""
    if len(shap_values.values.shape) == 3:
        base_values = shap_values.base_values
        if len(base_values.shape) == 2:
            selected_base_values = base_values[:, class_index]
        else:
            selected_base_values = base_values
        return shap.Explanation(
            values=shap_values.values[:, :, class_index],
            base_values=selected_base_values,
            data=shap_values.data,
            feature_names=shap_values.feature_names,
        )
    return shap_values


def generate_shap_visualizations(
    model: LogisticRegression,
    x_train: spmatrix,
    x_test: spmatrix,
    vectorizer: TfidfVectorizer,
    results_dir: str | Path = "results",
) -> None:
    """Compute SHAP values and save summary and waterfall plots to PNG files."""
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)

    x_train_sample = x_train[: min(500, x_train.shape[0])]
    x_test_sample = x_test[: min(200, x_test.shape[0])]

    feature_names = vectorizer.get_feature_names_out().tolist()
    explainer = shap.LinearExplainer(model, x_train_sample, feature_names=feature_names)
    shap_values = explainer(x_test_sample)
    class_specific_values = _slice_for_class(shap_values, class_index=0)
    dense_features = x_test_sample.toarray()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        class_specific_values.values,
        dense_features,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(destination / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    first_sample = shap.Explanation(
        values=class_specific_values.values[0],
        base_values=class_specific_values.base_values[0]
        if hasattr(class_specific_values.base_values, "__len__")
        else class_specific_values.base_values,
        data=dense_features[0],
        feature_names=feature_names,
    )
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(first_sample, show=False)
    plt.tight_layout()
    plt.savefig(destination / "shap_waterfall_sample0.png", dpi=300, bbox_inches="tight")
    plt.close()
